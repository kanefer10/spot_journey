import streamlit as st
# Required for Snowpark connection outside Snowflake environment
from snowflake.snowpark import Session
# Alias Snowpark min and max to avoid conflict with built-in functions
from snowflake.snowpark.functions import col, min as snowpark_min, max as snowpark_max, lit, lower
from snowflake.snowpark.exceptions import SnowparkSQLException
import pandas as pd
import logging
import re # Import regex for URI parsing
from streamlit.components.v1 import iframe # Import iframe component

# --- Set Page Config FIRST ---
# This MUST be the first Streamlit command executed
st.set_page_config(layout="wide")

# --- Configuration ---
BASE_TABLE_NAME = "song_class.public.song_data_table" # Use a constant
RESULTS_PER_PAGE = 20
NUMERIC_FILTER_COLUMNS = [
    "DANCEABILITY", "LOUDNESS", "SPEECHINESS", "LIVENESS", "VALENCE",
    "TEMPO", "ENERGY", "ACOUSTICNESS", "INSTRUMENTALNESS",
    "KEY", "MODE", "DURATION_MS", "TIME_SIGNATURE"
]
# Column for stable sorting (IMPORTANT for pagination!) - Use quoted if needed
ORDER_BY_COLUMN = "NAME" # Assuming "NAME" is suitable for sorting
TRACK_URI_COLUMN = "TRACK_URI" # Define the column name for Spotify URIs

# --- Setup Logging (Optional) ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions with Caching ---

# --- MODIFIED: Function to connect to Snowflake using st.secrets ---
@st.cache_resource(ttl=600) # Cache resource for 10 minutes
def connect_to_snowflake():
    """Connects to Snowflake using credentials from st.secrets"""
    logger.info("Attempting to connect to Snowflake using st.secrets...")
    try:
        # Check if secrets are loaded
        if "snowflake" not in st.secrets:
             st.error("Snowflake credentials not found in st.secrets. Please configure secrets in Streamlit Cloud.")
             st.stop()

        connection_parameters = st.secrets["snowflake"]
        session = Session.builder.configs(connection_parameters).create()
        logger.info("Snowflake session created successfully.")
        # Test connection with a simple query
        session.sql("SELECT 1").collect()
        logger.info("Snowflake session tested successfully.")
        return session
    except Exception as e:
        logger.error(f"Error connecting to Snowflake: {e}", exc_info=True)
        st.exception(e) # Show detailed error in app
        st.error(f"Error connecting to Snowflake. Check your credentials in st.secrets and Snowflake connection details. Error: {e}")
        st.stop()

# --- REMOVED get_active_session() function ---

@st.cache_data(ttl=3600) # Cache distinct stages for 1 hour
def get_distinct_stages_with_numbers(_session, table_name):
    """
    Fetches distinct stage names and numbers, ordered by stage number.
    Returns a list suitable for st.selectbox with format_func.
    Format: [(filter_value, display_value), ...] where filter_value is STAGE_NAME.
    Includes an "All" option prepended.
    """
    logger.info(f"Fetching distinct stage names and numbers from {table_name}...")
    # Default "All" option structure
    all_option = ("All", "All Stages") # (filter_value, display_value)

    try:
        # Attempt to select both number and name, ordering by number
        # Ensure column names ("STAGE_NUMBER", "STAGE_NAME") are correct for your table
        sql_query = f"""
            SELECT DISTINCT "STAGE_NUMBER", "STAGE_NAME"
            FROM {table_name}
            WHERE "STAGE_NAME" IS NOT NULL AND "STAGE_NUMBER" IS NOT NULL
            ORDER BY "STAGE_NUMBER" ASC
        """
        logger.info(f"Executing SQL for distinct stages: {sql_query}")
        stage_df = _session.sql(sql_query).to_pandas()

        # Process the dataframe to create the list of tuples
        # Handle potential non-numeric stage numbers gracefully
        stage_list = []
        for _, row in stage_df.iterrows():
            stage_name = row["STAGE_NAME"]
            try:
                # Attempt to convert stage number to int for clean display
                stage_num = int(row["STAGE_NUMBER"])
                display_value = f"{stage_num} - {stage_name}"
            except (ValueError, TypeError):
                # Fallback if stage number isn't a clean integer
                stage_num_raw = row["STAGE_NUMBER"]
                display_value = f"{stage_num_raw} - {stage_name}"
                logger.warning(f"Could not convert STAGE_NUMBER '{stage_num_raw}' to int for stage '{stage_name}'. Using raw value.")

            stage_list.append((stage_name, display_value)) # (filter_value, display_value)

        logger.info(f"Fetched {len(stage_list)} distinct stages.")
        # Prepend the "All" option
        return [all_option] + stage_list

    except SnowparkSQLException as e:
        # Log the specific SQL error
        logger.error(f"Snowpark SQL error fetching stages: {e}", exc_info=True)
        if "order by expression" in str(e) or "STAGE_NUMBER" in str(e):
             st.warning(f"Could not order stages by STAGE_NUMBER (DB error): {e}. Check column name/permissions. Falling back to alphabetical sort by name.")
             # Fallback: Fetch only names and sort alphabetically
             return get_distinct_stages_fallback(_session, table_name)
        else:
             st.warning(f"Could not fetch stages (DB error): {e}. Check query/permissions.")
             return [all_option] # Return only "All" on other SQL errors
    except Exception as e:
        logger.error(f"Unexpected error fetching stages: {e}", exc_info=True)
        st.warning(f"Could not fetch stages (unexpected error): {e}")
        return [all_option] # Return only "All"

# Fallback function if ordering by STAGE_NUMBER fails
def get_distinct_stages_fallback(_session, table_name):
    """Fetches only distinct stage names, sorted alphabetically."""
    logger.warning("Using fallback to fetch only stage names.")
    all_option = ("All", "All Stages")
    try:
        sql_query = f"""
            SELECT DISTINCT "STAGE_NAME"
            FROM {table_name}
            WHERE "STAGE_NAME" IS NOT NULL
        """
        stage_df = _session.sql(sql_query).to_pandas()
        # Sort alphabetically
        stage_list_names = sorted(stage_df["STAGE_NAME"].dropna().unique().tolist())
        # Format as (filter_value, display_value) - same value for both in this case
        stage_list = [(name, name) for name in stage_list_names]
        logger.info(f"Fetched {len(stage_list)} distinct stages (fallback).")
        return [all_option] + stage_list
    except Exception as e:
        logger.error(f"Error in fallback stage fetching: {e}", exc_info=True)
        st.warning("Could not fetch stage names even with fallback.")
        return [all_option]


@st.cache_data(ttl=3600) # Cache numeric bounds for 1 hour
def get_numeric_column_bounds(_session, table_name, columns):
    """Fetches min/max for multiple numeric columns in one query."""
    logger.info(f"Fetching numeric bounds for {len(columns)} columns from {table_name}...")
    if not columns:
        return {}
    try:
        agg_exprs = []
        valid_columns_for_agg = []
        # Check column existence/type conceptually before building query if possible
        # (Actual check happens in Snowflake)
        for col_name in columns:
            # Use aliased Snowpark functions: snowpark_min and snowpark_max
            # Use quoted identifiers if needed: col(f'"{col_name}"')
            agg_exprs.append(snowpark_min(col(col_name)).alias(f"MIN_{col_name}"))
            agg_exprs.append(snowpark_max(col(col_name)).alias(f"MAX_{col_name}"))
            valid_columns_for_agg.append(col_name) # Keep track of columns included

        if not agg_exprs:
             logger.warning("No valid columns found for numeric bounds aggregation.")
             return {}

        stats_df = _session.table(table_name).agg(*agg_exprs).to_pandas()

        if not stats_df.empty:
            bounds = stats_df.iloc[0].to_dict()
            # Basic validation of bounds
            valid_bounds = {}
            for col_name in valid_columns_for_agg: # Iterate only over columns included in agg
                min_key, max_key = f"MIN_{col_name}", f"MAX_{col_name}"
                if min_key in bounds and max_key in bounds and \
                   pd.notna(bounds[min_key]) and pd.notna(bounds[max_key]):
                   try:
                       min_f, max_f = float(bounds[min_key]), float(bounds[max_key])
                       if min_f <= max_f:
                           valid_bounds[min_key] = min_f
                           valid_bounds[max_key] = max_f
                       else:
                            # Log but maybe allow slider with swapped min/max or default range?
                            logger.warning(f"Min > Max for {col_name} ({min_f} > {max_f}). Using raw bounds.")
                            valid_bounds[min_key] = min_f # Keep original bounds for now
                            valid_bounds[max_key] = max_f
                   except (ValueError, TypeError) as e:
                       logger.warning(f"Could not convert bounds for {col_name} to float. Skipping. Error: {e}")
                else:
                    logger.warning(f"Missing or invalid bounds returned for {col_name}. Skipping.")

            logger.info(f"Successfully fetched and validated numeric bounds for {len(valid_bounds)//2} columns.")
            return valid_bounds
        else:
            logger.warning("Numeric bounds query returned empty results.")
            return {}
    except SnowparkSQLException as e:
         logger.error(f"Snowpark SQL error fetching numeric bounds: {e}", exc_info=True)
         # Check if error is due to non-numeric column type
         if "numeric value" in str(e).lower() or "data type" in str(e).lower():
              st.warning(f"Could not fetch numeric bounds. Ensure all filter columns ({', '.join(columns)}) are numeric. Error: {e}")
         else:
              st.warning(f"Could not fetch numeric bounds (DB error): {e}. Check column names and permissions.")
         return {}
    except Exception as e:
        logger.error(f"Unexpected error fetching numeric bounds: {e}", exc_info=True)
        st.warning(f"Could not fetch numeric bounds (unexpected error): {e}")
        return {}

# --- Spotify Helper ---
def get_track_id_from_uri(uri):
    """Extracts the Spotify Track ID from a URI string."""
    if not uri or not isinstance(uri, str):
        return None
    # Updated regex to handle potential variations or query parameters
    match = re.search(r"spotify:track:([a-zA-Z0-9]+)(?:[?].*)?$", uri)
    if match:
        return match.group(1)
    logger.warning(f"Could not parse track ID from URI: {uri}")
    return None

# --- Main App ---

# --- MODIFIED: Connect using the new function ---
session = connect_to_snowflake()

# Initialize session state for the selected track URI
if 'selected_track_uri' not in st.session_state:
    st.session_state.selected_track_uri = None
# Initialize state to store the dataframe for the current page
if 'current_page_df' not in st.session_state:
    st.session_state.current_page_df = pd.DataFrame()


# Set up the rest of the UI
st.title("🎵 Songs Explorer")
st.markdown("Use filters in the sidebar to explore songs and their features. Select a song below the table to play a preview.")

# --- Spotify Player Area (Placeholder - will be filled later) ---
spotify_player_area = st.empty()

# --- Placeholder for Song Selection ---
song_selector_area = st.empty()

# --- Sidebar Filters ---
selected_stage_filter_value = None # Initialize variable to store the actual filter value (STAGE_NAME)
slider_filters = {} # Initialize dictionary

with st.sidebar:
    st.header("Filters")

    # --- Stage Name Filter (Moved to Top) ---
    # Pass the established session object to the helper function
    stage_options_data = get_distinct_stages_with_numbers(session, BASE_TABLE_NAME)
    # Only show selectbox if options were successfully fetched (more than just the "All" option)
    if len(stage_options_data) > 1:
        # Use format_func to display the second element (display_value) of the tuple
        selected_stage_filter_value = st.selectbox(
            "Stage Name",
            options=stage_options_data, # Provide the list of (filter_value, display_value) tuples
            format_func=lambda option: option[1], # Function to display the second element
            index=0, # Default to the first option ("All")
            key="stage_select"
        )[0] # Get the first element (filter_value) which is the actual STAGE_NAME
    else:
        # Warning is already shown in the helper function if fetching failed
        st.info("Stage Name filter unavailable.")


    # --- Search by track name (NAME column) - Case-insensitive ---
    track_search = st.text_input("Search by Track Name (contains)", key="track_search_input") # Added key


    # --- Sliders for numerical filters within an expander ---
    st.markdown("---") # Separator
    with st.expander("Numeric Filters", expanded=False): # Group sliders, start collapsed
        # Pass the established session object to the helper function
        numeric_bounds = get_numeric_column_bounds(session, BASE_TABLE_NAME, NUMERIC_FILTER_COLUMNS)

        for col_name in NUMERIC_FILTER_COLUMNS:
            min_key = f"MIN_{col_name}"
            max_key = f"MAX_{col_name}"

            # Check if valid bounds were retrieved for this column
            if min_key in numeric_bounds and max_key in numeric_bounds:
                min_f = numeric_bounds[min_key]
                max_f = numeric_bounds[max_key]

                # Handle potential min > max from warning log above
                if min_f > max_f:
                    st.warning(f"Data issue for {col_name}: Min value ({min_f}) > Max value ({max_f}). Slider range might be incorrect.")
                    # Using swapped values for slider range
                    min_slider, max_slider = max_f, min_f
                else:
                     min_slider, max_slider = min_f, max_f

                default_value = (min_slider, max_slider) # Default is the full (potentially corrected) range

                try:
                    # Create the slider with a unique key
                    slider_key = f"slider_{col_name}"
                    selected_range = st.slider(
                        label=col_name.replace("_", " ").capitalize(), # Prettier label
                        min_value=min_slider,
                        max_value=max_slider,
                        value=default_value, # Start with full range
                        key=slider_key # Add key for potential reset later
                    )

                    # Store the selected range *only if it's different* from the full range
                    # Compare against the actual min/max used for the slider
                    if selected_range != default_value:
                        # Store the selected range from the slider
                        slider_filters[col_name] = selected_range
                except Exception as slider_error:
                     # Catch errors during slider creation (e.g., if min/max are still problematic)
                     logger.error(f"Error creating slider for {col_name}: {slider_error}", exc_info=True)
                     st.warning(f"Could not create filter slider for {col_name}.")

            else:
                # Warning is already shown in get_numeric_column_bounds if bounds failed
                pass # Optionally add st.info(f"Filter for {col_name} unavailable.")


# --- Apply Filters ---
# Start with the base DataFrame using the established session
filtered_df = session.table(BASE_TABLE_NAME)

# Apply stage name filter using the selected filter value
if selected_stage_filter_value is not None and selected_stage_filter_value != "All":
    try:
        # Use quoted identifier if needed: col('"STAGE_NAME"')
        filtered_df = filtered_df.filter(col("STAGE_NAME") == selected_stage_filter_value)
        logger.info(f"Applied stage filter: {selected_stage_filter_value}")
    except Exception as e:
        logger.error(f"Error applying stage filter: {e}", exc_info=True)
        st.warning("Could not apply stage filter.")


# Apply track name search filter (case-insensitive using lower)
if track_search:
    try:
        # Ensure lower function is available (imported at the top)
        filtered_df = filtered_df.filter(lower(col("NAME")).like(f"%{track_search.lower()}%"))
        logger.info(f"Applied track search: {track_search}")
    except Exception as e:
         logger.error(f"Error applying track search filter: {e}", exc_info=True)
         st.warning("Could not apply track search filter.")


# Apply numeric slider filters
if slider_filters:
    logger.info(f"Applying {len(slider_filters)} numeric filters: {slider_filters}")
    for col_name, (min_val, max_val) in slider_filters.items():
        try:
            # Use lit() for literal values
            # Use quoted identifier if needed: col(f'"{col_name}"')
            filtered_df = filtered_df.filter((col(col_name) >= lit(min_val)) & (col(col_name) <= lit(max_val)))
        except Exception as e:
             logger.error(f"Error applying numeric filter for {col_name}: {e}", exc_info=True)
             st.warning(f"Could not apply filter for {col_name}.")


# --- Pagination ---
# Count rows *after* all filters are applied - ADD SPINNER
row_count = 0 # Initialize row_count
try:
    logger.info("Counting total rows after filtering...")
    with st.spinner("Counting matching songs..."): # Add spinner
        # This executes the query plan built so far to get the count
        row_count = filtered_df.count()
    logger.info(f"Total rows found: {row_count}")
except SnowparkSQLException as e:
    logger.error(f"Snowpark SQL error counting rows: {e}", exc_info=True)
    st.error(f"Error counting results: {e}. Filters might be invalid or incompatible.")
except Exception as e:
    logger.error(f"Unexpected error counting rows: {e}", exc_info=True)
    st.error(f"Unexpected error counting results: {e}")

# Calculate total pages
if row_count > 0:
    total_pages = (row_count + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
else:
    total_pages = 1 # At least one page

# Pagination controls in the main area
st.markdown("---") # Separator before pagination/results
col1, col2 = st.columns([1, 3]) # Layout columns for pagination controls

with col1:
    # Ensure max_value is at least 1
    # The built-in `max` function is now used correctly due to the aliased import
    current_page = st.number_input(
        "Page",
        min_value=1,
        max_value=max(1, total_pages), # Use Python's built-in max here
        value=1,
        step=1,
        key="page_number", # Add a key for stability
        help=f"Select page number (1 to {total_pages})"
    )

with col2:
    # Display count info, handle case where count failed (row_count=0)
     st.caption(f"Page {current_page} of {total_pages} | Total matching songs: {row_count:,}")


# Calculate offset for the current page
offset = (current_page - 1) * RESULTS_PER_PAGE


# --- Display Spotify Player ---
# This runs *before* displaying the song list, using the state from the *previous* run
if st.session_state.selected_track_uri:
    track_id = get_track_id_from_uri(st.session_state.selected_track_uri)
    if track_id:
        # --- CORRECTED EMBED URL ---
        embed_url = f"https://open.spotify.com/embed/track/{track_id}"
        # Use the placeholder to embed the iframe
        with spotify_player_area.container():
             st.markdown("#### Now Playing Preview")
             iframe(embed_url, height=152) # Use standard height for better UI
             st.markdown("---") # Add separator after player
    else:
        # Clear the area if URI is invalid
        spotify_player_area.empty()
else:
    # Clear the area if no track is selected
    spotify_player_area.empty()


# --- Fetch and Display Data Table ---
# Only attempt to display data if the count was successful and > 0
if row_count > 0:
    try:
        # Define columns to display - REORDERED
        # Get schema from the *filtered* dataframe before applying pagination
        all_available_cols = [c.name for c in filtered_df.schema]

        # 1. Start with fixed columns if they exist
        display_order = []
        if "NAME" in all_available_cols:
            display_order.append("NAME")
        if "ARTISTS" in all_available_cols:
            display_order.append("ARTISTS")

        # 2. Add available numeric columns (from NUMERIC_FILTER_COLUMNS) that haven't been added yet
        numeric_cols_in_schema = [
            col for col in NUMERIC_FILTER_COLUMNS
            if col in all_available_cols and col not in display_order
        ]
        display_order.extend(sorted(numeric_cols_in_schema)) # Sort numeric columns alphabetically

        # 3. Add remaining columns (that are not already included), sorted alphabetically
        processed_cols = set(display_order) # Keep track of columns already added
        other_cols = [
            col for col in all_available_cols
            if col not in processed_cols and col != TRACK_URI_COLUMN # Exclude URI from display table
        ]
        display_order.extend(sorted(other_cols))

        # This is the final order for display
        display_columns = display_order

        # Ensure the ORDER_BY_COLUMN and TRACK_URI_COLUMN are selected for sorting/functionality
        select_cols_for_query = list(display_columns) # Start with display columns
        # Add TRACK_URI if it exists and isn't already included
        if TRACK_URI_COLUMN in all_available_cols and TRACK_URI_COLUMN not in select_cols_for_query:
            select_cols_for_query.append(TRACK_URI_COLUMN)
        # Add ORDER_BY column if it exists and isn't already included
        if ORDER_BY_COLUMN in all_available_cols and ORDER_BY_COLUMN not in select_cols_for_query:
             select_cols_for_query.append(ORDER_BY_COLUMN)
        elif ORDER_BY_COLUMN not in all_available_cols:
             logger.warning(f"Order by column '{ORDER_BY_COLUMN}' not found in DataFrame schema. Pagination might be inconsistent.")


        # ADD SPINNER for data fetching/processing
        with st.spinner(f"Loading page {current_page}..."):
            # Select columns, Order data, Apply limit and offset
            # **MODIFIED PAGINATION APPROACH**
            # Fetch limit + offset rows, then slice in Pandas
            rows_to_fetch = offset + RESULTS_PER_PAGE
            logger.info(f"Fetching data up to row {rows_to_fetch} (for page {current_page}) ordered by {ORDER_BY_COLUMN}")

            # Build the query: select -> order -> limit (fetching potentially more rows)
            # Select only the columns needed for the query (includes sort and URI columns)
            paginated_query = filtered_df.select(select_cols_for_query)

            # Apply ordering only if the column exists in the selected columns
            if ORDER_BY_COLUMN in select_cols_for_query:
                 # Use quoted identifier for order_by if needed: col(f'"{ORDER_BY_COLUMN}"')
                 paginated_query = paginated_query.order_by(col(ORDER_BY_COLUMN).asc())
            # Apply limit to fetch necessary rows up to the end of the current page
            paginated_query = paginated_query.limit(rows_to_fetch) # REMOVED .offset()


            # Fetch data to Pandas - this executes the query
            all_fetched_pandas_df = paginated_query.to_pandas()
            logger.info(f"Successfully fetched {len(all_fetched_pandas_df)} rows total for slicing.")

            # Slice the Pandas DataFrame to get the rows for the current page
            display_pandas_df = all_fetched_pandas_df.iloc[offset:]
            # Store the current page's dataframe in session state for the selector
            st.session_state.current_page_df = display_pandas_df

        # --- Display using st.dataframe ---
        # Ensure display_columns only contains columns present in the fetched pandas df
        final_display_columns = [col for col in display_columns if col in display_pandas_df.columns]
        st.dataframe(display_pandas_df[final_display_columns], use_container_width=True, hide_index=True) # Reverted to dataframe

        # --- Song Selector for Playback ---
        with song_selector_area.container():
            st.markdown("---")
            st.markdown("#### Select Song to Play Preview")
            # Create options for the selectbox: "Track Name by Artist" (Index)
            # Filter out rows with invalid URIs first
            playable_songs_df = st.session_state.current_page_df[
                st.session_state.current_page_df[TRACK_URI_COLUMN].apply(lambda x: bool(get_track_id_from_uri(x)))
            ].copy() # Create a copy to avoid SettingWithCopyWarning

            if not playable_songs_df.empty:
                # Create display labels, using index as a unique identifier within the page
                playable_songs_df['display_label'] = playable_songs_df.apply(
                    lambda row: f"{row.get('NAME', 'Unknown Track')} by {row.get('ARTISTS', 'Unknown Artist')} (Index: {row.name})",
                    axis=1
                )
                song_options = pd.Series(playable_songs_df.index, index=playable_songs_df['display_label']).to_dict()

                # Add a "None" option
                options_with_none = {"-- Select a Song --": None}
                options_with_none.update(song_options)

                selected_song_label = st.selectbox(
                    "Select Song:",
                    options=options_with_none.keys(),
                    index=0, # Default to "-- Select a Song --"
                    key="song_selector"
                )

                # Get the corresponding index from the selected label
                selected_index = options_with_none[selected_song_label]

                if selected_index is not None:
                    # Get the URI from the original dataframe using the selected index
                    selected_uri = st.session_state.current_page_df.loc[selected_index, TRACK_URI_COLUMN]
                    # Update session state only if the selection changed
                    if st.session_state.selected_track_uri != selected_uri:
                        st.session_state.selected_track_uri = selected_uri
                        st.rerun() # Rerun to update the player
                # If "-- Select a Song --" is chosen and a track is playing, clear it
                elif st.session_state.selected_track_uri is not None:
                     st.session_state.selected_track_uri = None
                     st.rerun()

            else:
                 st.info("No playable songs with valid Spotify URIs on this page.")


    except SnowparkSQLException as e:
        logger.error(f"Snowpark SQL error displaying data: {e}", exc_info=True)
        st.error(f"Error fetching data for display: {e}. Check query logic or permissions.")
        st.dataframe(pd.DataFrame()) # Show empty dataframe on error
    except Exception as e:
        logger.error(f"Unexpected error displaying data: {e}", exc_info=True)
        st.error(f"Unexpected error displaying data: {e}")
        st.dataframe(pd.DataFrame()) # Show empty dataframe on error
elif row_count == 0:
    # Display message if no rows match filters (and count was successful)
    st.info("No songs match the current filters. Try adjusting the filters in the sidebar.") # Slightly improved message
    # Clear the current page df and selected track if no results
    st.session_state.current_page_df = pd.DataFrame()
    if st.session_state.selected_track_uri is not None:
        st.session_state.selected_track_uri = None
        st.rerun()

# Else: an error occurred during count, message already shown
