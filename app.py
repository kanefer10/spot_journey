# -*- coding: utf-8 -*-
"""
Streamlit app to explore and shuffle songs classified by stage,
querying data directly from a Snowflake table using Streamlit in Snowflake.
Reads configuration AND stage specs from config.yaml file.
Includes comprehensive filtering, pagination below tables, an all catalog view,
improved shuffle logic, UI enhancements, a Spec Tester tab,
and an embedded Spotify player loaded using st.data_editor interaction.
Uses sidebar radio for navigation. Access controlled by Snowflake roles.
"""

import streamlit as st
# import duckdb # Removed duckdb
import pandas as pd
import os
import random
import math
import yaml # Import YAML library
import time # For debugging timestamp
import csv # Import csv library again
import re # For extracting track ID
import json # For parsing service account key
from snowflake.snowpark.context import get_active_session # Added Snowpark session
# Removed GCS/AWS/HF specific libraries

# --- Page Configuration (MUST be the first Streamlit command after imports) ---
st.set_page_config(layout="wide", page_title="Sound Journey Explorer")

# --- REMOVED Password Protection ---
# def check_password(): ...
# if not check_password(): st.stop()

# --- Load Configuration ---
CONFIG_FILE = 'config.yaml'
DEFINITIONS_FILE = 'Spotify_Data_Dictionary.csv' # Keep for definitions

# *** Define Snowflake Table Name ***
# *** Replace with your actual Database.Schema.Table name if different ***
SNOWFLAKE_DATA_TABLE = "SONG_DATA.PUBLIC.SONG_DATA_TABLE"

@st.cache_data(ttl=30)
def load_config():
    """Loads configuration and stage specs from the YAML file."""
    if not os.path.exists(CONFIG_FILE): return None, f"Config file '{CONFIG_FILE}' not found."
    try:
        with open(CONFIG_FILE, 'r') as f: config_data = yaml.safe_load(f)
        # Only need stage_specs from config now
        if not config_data or 'stage_specs' not in config_data:
             return None, f"Config file '{CONFIG_FILE}' missing required key 'stage_specs'."
        # Remove validation for local_data_path
        return config_data, None
    except yaml.YAMLError as e: return None, f"Error parsing config file '{CONFIG_FILE}': {e}"
    except Exception as e: return None, f"Error reading config file '{CONFIG_FILE}': {e}"

app_config, config_error_msg = load_config()

@st.cache_data(ttl=3600)
def load_definitions(filepath):
    """Loads feature definitions from the CSV file, trying different encodings."""
    definitions = {}
    # Check relative path first, then absolute if needed (for SiS stage)
    if not os.path.exists(filepath) and not os.path.isabs(filepath):
        # Try constructing path relative to script if running in SiS/stage
        script_dir = os.path.dirname(__file__)
        filepath_alt = os.path.join(script_dir, filepath)
        if not os.path.exists(filepath_alt):
             st.warning(f"Definitions file '{filepath}' not found in current dir or script dir.")
             return definitions
        else:
             filepath = filepath_alt # Use path relative to script

    encodings_to_try = ['utf-8', 'latin-1', 'windows-1252']
    for encoding in encodings_to_try:
        try:
            with open(filepath, mode='r', encoding=encoding) as infile:
                if encoding == 'utf-8' and infile.read(1) != '\ufeff': infile.seek(0)
                reader = csv.DictReader(infile); current_definitions = {}
                for row in reader:
                    if 'variable' in row and 'description' in row: current_definitions[row['variable'].lower()] = row['description'].strip()
                print(f"Definitions loaded successfully using {encoding}."); return current_definitions
        except UnicodeDecodeError: print(f"Decoding with {encoding} failed, trying next..."); continue
        except Exception as e: st.warning(f"Error reading definitions file '{filepath}' with {encoding}: {e}"); return {}
    st.error(f"Could not read definitions file '{filepath}' with any attempted encoding."); return {}

definitions = load_definitions(DEFINITIONS_FILE)

# --- App Configuration ---
if app_config is None: st.error(config_error_msg); st.error("Stopping execution."); st.stop()

# Removed data path and file format config
DEFAULT_SONGS_PER_PAGE = app_config.get('songs_per_page', 50)
DEFAULT_MAX_SCORE = app_config.get('default_max_score_filter', 100.0)
STAGE_SPECS = app_config.get('stage_specs', {})

if STAGE_SPECS:
    first_stage_data = next(iter(STAGE_SPECS.values()))
    ALL_FEATURES_FROM_SPEC = [k for k in first_stage_data if k != 'stage_number']
else: ALL_FEATURES_FROM_SPEC = []; st.warning("Stage specifications seem empty in config.yaml.")

CONTINUOUS_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'total_mismatch_score']
DISCRETE_FEATURES = ['key']
ALL_AUDIO_FEATURES = ['key', 'mode', 'time_signature', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
FEATURES_FOR_UI = sorted(list(set(ALL_FEATURES_FROM_SPEC + DISCRETE_FEATURES + ['total_mismatch_score', 'stage_rank'])))

ROWS_PER_PAGE_OPTIONS = [25, 50, 100, 200]
if DEFAULT_SONGS_PER_PAGE not in ROWS_PER_PAGE_OPTIONS: ROWS_PER_PAGE_OPTIONS.append(DEFAULT_SONGS_PER_PAGE); ROWS_PER_PAGE_OPTIONS.sort()

VIEWS = ["Stage Explorer", "All Catalog", "Shuffle Journey", "Spec Tester"]
VIEW_ICONS = ["🔍", "🎶", "🔀", "🧪"]

DEFAULT_TRACK_URI = "spotify:track:7GhIk7Il098yCjg4BQjzvb"

# --- End Configuration ---

# --- Initialize Session State ---
# ... (keep existing initializations) ...
if 'explorer_page' not in st.session_state: st.session_state.explorer_page = 1
if 'catalog_page' not in st.session_state: st.session_state.catalog_page = 1
if 'explorer_rows_per_page' not in st.session_state: st.session_state.explorer_rows_per_page = DEFAULT_SONGS_PER_PAGE
if 'catalog_rows_per_page' not in st.session_state: st.session_state.catalog_rows_per_page = DEFAULT_SONGS_PER_PAGE
if 'test_song_id' not in st.session_state: st.session_state.test_song_id = ""
if 'shuffle_df' not in st.session_state: st.session_state.shuffle_df = None
if 'shuffle_fallback' not in st.session_state: st.session_state.shuffle_fallback = False
if 'selected_track_uri' not in st.session_state: st.session_state.selected_track_uri = DEFAULT_TRACK_URI
if 'current_view' not in st.session_state: st.session_state.current_view = VIEWS[0]


# --- Snowflake Session ---
# Get the active Snowpark session
try:
    session = get_active_session()
    print("Successfully obtained Snowflake session.")
except Exception as e:
    # Fallback for local testing (requires snowflake.connector and credentials)
    # For simplicity, we'll just error out if not in SiS for now.
    st.error(f"Could not get Snowflake session: {e}")
    st.info("This app is intended to be run from within Snowflake.")
    st.stop() # Stop if session cannot be obtained

# --- Data Loading and Querying Functions (Using Snowflake) ---

def get_stage_mapping():
    mapping = {}
    for name, details in STAGE_SPECS.items():
        num = details.get('stage_number')
        if num is not None: mapping[num] = name
        else: print(f"Warning: Missing 'stage_number' for stage '{name}' in config.")
    return dict(sorted(mapping.items()))

@st.cache_data(ttl=3600)
def get_feature_ranges_snowflake(_session, features):
    """Gets min/max values for features from Snowflake table."""
    if not _session: return {}
    ranges = {}
    table_name = SNOWFLAKE_DATA_TABLE
    try:
        schema_df = _session.sql(f"DESCRIBE TABLE {table_name}").to_pandas()
        available_columns_schema = schema_df['name'].tolist() # Keep original case for query
        available_columns_lower = [c.lower() for c in available_columns_schema]
    except Exception as e:
        st.warning(f"Could not describe table {table_name} to get feature ranges: {e}")
        available_columns_lower = [f.lower() for f in features] # Assume all exist on error
        available_columns_schema = features # Use original case as fallback

    features_to_query = [f for f in features if f.lower() in available_columns_lower]
    if not features_to_query: st.warning("Could not find expected feature columns in data table."); return {feat: (0.0, 100.0) if feat == 'total_mismatch_score' else (0.0, 1.0) for feat in features}

    select_clauses = []
    for feat in features_to_query:
         actual_col_name = next((c for c in available_columns_schema if c.lower() == feat.lower()), feat) # Use original case
         select_clauses.append(f"MIN(\"{actual_col_name}\") as min_{feat.lower()}")
         select_clauses.append(f"MAX(\"{actual_col_name}\") as max_{feat.lower()}")

    if not select_clauses: return {feat: (0.0, 100.0) if feat == 'total_mismatch_score' else (0.0, 1.0) for feat in features}

    query = f"SELECT {', '.join(select_clauses)} FROM {table_name}"
    try:
        result_df = _session.sql(query).to_pandas()
        if not result_df.empty:
            row = result_df.iloc[0]
            for feat in features:
                min_val = row.get(f'min_{feat.lower()}', None); max_val = row.get(f'max_{feat.lower()}', None)
                num_min_val = pd.to_numeric(min_val, errors='coerce'); num_max_val = pd.to_numeric(max_val, errors='coerce')
                spec_min, spec_max = 0.0, 1.0
                if feat == 'total_mismatch_score': spec_min, spec_max = 0.0, DEFAULT_MAX_SCORE
                all_spec_vals = []
                for stage_detail in STAGE_SPECS.values():
                     if feat in stage_detail and isinstance(stage_detail.get(feat), list) and len(stage_detail[feat]) == 2: all_spec_vals.extend(stage_detail[feat])
                if all_spec_vals:
                     try: numeric_spec_vals = [float(v) for v in all_spec_vals]; spec_min, spec_max = min(numeric_spec_vals), max(numeric_spec_vals)
                     except (ValueError, TypeError): pass
                if pd.isna(num_min_val): num_min_val = spec_min
                if pd.isna(num_max_val): num_max_val = spec_max
                ranges[feat] = (num_min_val, num_max_val)
        if 'total_mismatch_score' not in ranges: ranges['total_mismatch_score'] = (0.0, DEFAULT_MAX_SCORE)
        return ranges
    except Exception as e:
        st.warning(f"Could not fetch feature ranges from Snowflake: {e}")
        return {feat: (0.0, DEFAULT_MAX_SCORE) if feat == 'total_mismatch_score' else (0.0, 1.0) for feat in features}

@st.cache_data(ttl=3600)
def get_discrete_feature_values_snowflake(_session, features):
    """Gets unique values for discrete features from Snowflake table."""
    if not _session: return {}
    values = {}
    table_name = SNOWFLAKE_DATA_TABLE
    try:
        schema_df = _session.sql(f"DESCRIBE TABLE {table_name}").to_pandas()
        available_columns_schema = schema_df['name'].tolist()
        available_columns_lower = [c.lower() for c in available_columns_schema]
    except Exception as e:
        st.warning(f"Could not describe table {table_name} for discrete values: {e}")
        available_columns_lower = [f.lower() for f in features] # Assume columns exist
        available_columns_schema = features

    try:
        for feat in features:
            if feat.lower() in available_columns_lower:
                actual_col_name = next((c for c in available_columns_schema if c.lower() == feat.lower()), feat) # Fallback
                query = f"SELECT DISTINCT \"{actual_col_name}\" FROM {table_name} WHERE \"{actual_col_name}\" IS NOT NULL ORDER BY \"{actual_col_name}\""
                result = _session.sql(query).to_pandas()
                values[feat] = sorted(result[actual_col_name].tolist()) if not result.empty else []
            else:
                print(f"Warning: Discrete feature '{feat}' not found in table {table_name}.")
                values[feat] = []
        return values
    except Exception as e:
        st.warning(f"Could not fetch discrete feature values for {feat} from Snowflake: {e}"); return {feat: [] for feat in features}

def build_where_clause_snowflake(stage_num, filters, feature_ranges_local):
    """Builds the WHERE clause for Snowflake SQL."""
    where_clauses = []
    params = {} # Use parameters for safety

    if stage_num is not None:
        where_clauses.append("stage_number = :stage_num_param")
        params["stage_num_param"] = stage_num

    if stage_num is not None and 'score' in filters and filters['score'] is not None:
        score_max_val = feature_ranges_local.get('total_mismatch_score', (0.0, DEFAULT_MAX_SCORE))[1]
        effective_max_score = min(float(filters['score']), float(score_max_val), 1000.0)
        where_clauses.append("total_mismatch_score <= :max_score_param")
        params["max_score_param"] = effective_max_score

    if 'artist' in filters and filters['artist']:
        where_clauses.append("lower(artists) LIKE lower(:artist_param)")
        params["artist_param"] = f"%{filters['artist']}%" # Add wildcards for LIKE
    if 'song' in filters and filters['song']:
        # Use correct column name 'name' before rename
        where_clauses.append("lower(name) LIKE lower(:song_param)")
        params["song_param"] = f"%{filters['song']}%"

    for feat in CONTINUOUS_FEATURES:
        if feat == 'total_mismatch_score': continue
        if feat in filters and filters[feat] is not None:
            min_val, max_val = filters[feat]
            default_min, default_max = feature_ranges_local.get(feat, (0,1))
            if min_val is not None and max_val is not None and (min_val > default_min or max_val < default_max):
                 param_min_key = f"{feat}_min"
                 param_max_key = f"{feat}_max"
                 # Quote feature names in case they are reserved words or contain special chars
                 where_clauses.append(f"\"{feat}\" >= :{param_min_key} AND \"{feat}\" <= :{param_max_key}")
                 params[param_min_key] = min_val
                 params[param_max_key] = max_val

    for feat in DISCRETE_FEATURES: # Only 'key'
        if feat in filters and filters[feat]:
            if filters[feat]: # Only add if list is not empty
                # Use Snowflake's ARRAY_CONTAINS or similar for list parameters if possible,
                # otherwise format list carefully (ensure correct quoting/escaping)
                # Using simple IN for now, assuming safe values
                formatted_values = []
                for v in filters[feat]:
                    if isinstance(v, str):
                        formatted_values.append(f"'{v.replace('\'', '\'\'')}'") # Quote and escape
                    elif isinstance(v, (int, float)):
                        formatted_values.append(str(v))
                if formatted_values:
                     where_clauses.append(f"\"{feat}\" IN ({', '.join(formatted_values)})")

    where_clause_str = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    return where_clause_str, params


@st.cache_data(ttl=600)
def count_songs_snowflake(_session, stage_num, filters_tuple, feature_ranges_local):
    """Counts songs in Snowflake table based on filters."""
    filters = dict(filters_tuple)
    if not _session: return 0
    where_clause_str, params = build_where_clause_snowflake(stage_num, filters, feature_ranges_local)
    # Use 'id' which is assumed to be the column name in the Snowflake table
    query = f"SELECT COUNT(DISTINCT id) as total_count FROM {SNOWFLAKE_DATA_TABLE} {where_clause_str}"
    try:
        # Use params argument for binding
        result = _session.sql(query, params=params).collect()
        # Snowflake returns column names in uppercase by default
        return result[0]['TOTAL_COUNT'] if result else 0
    except Exception as e:
        st.error(f"Error counting songs in Snowflake: {e}"); st.error(f"Query attempted: {query} with params: {params}"); return 0

def query_paged_songs_snowflake(_session, stage_num, filters, limit, offset, feature_ranges_local):
    """Queries paged songs from Snowflake table."""
    if not _session: return pd.DataFrame()
    where_clause_str, params = build_where_clause_snowflake(stage_num, filters, feature_ranges_local)
    # Define columns to select (use names from the Snowflake table)
    select_cols = ['id', 'name', 'artists', 'album', 'track_uri', 'stage_number', 'stage_name', 'stage_rank', 'total_mismatch_score'] + ALL_AUDIO_FEATURES
    # Filter to columns that likely exist based on ALL_AUDIO_FEATURES + base columns
    # This assumes the Snowflake table schema matches these reasonably well
    select_list_str = ", ".join([f'"{col}"' for col in select_cols])

    # Use id for ordering
    query = f"""
        SELECT {select_list_str} FROM {SNOWFLAKE_DATA_TABLE}
        {where_clause_str}
        ORDER BY CASE WHEN stage_number IS NOT NULL THEN stage_rank END ASC, total_mismatch_score ASC, id ASC
        LIMIT {limit} OFFSET {offset};
    """
    try:
        results_df = _session.sql(query, params=params).to_pandas()
        # Rename columns to match expected Python names (lowercase, song_id/name)
        results_df.columns = results_df.columns.str.lower() # Lowercase all columns first
        results_df = results_df.rename(columns={"name": "song_name", "id": "song_id"})
        return results_df
    except Exception as e:
        st.error(f"Error querying paged data from Snowflake: {e}"); st.error(f"Query attempted: {query} with params: {params}"); return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_candidate_songs_snowflake(_session, rank_threshold=1):
    """Loads potential candidate songs (rank 1) from Snowflake table."""
    if not _session: return pd.DataFrame()
    print(f"[{time.time()}] Loading candidate songs (Rank <= {rank_threshold}) from Snowflake...")
    # Select necessary columns for shuffle
    cols_for_shuffle = ['id', 'name', 'artists', 'album', 'track_uri', 'stage_number', 'stage_name', 'stage_rank', 'total_mismatch_score']
    select_list_str = ", ".join([f'"{col}"' for col in cols_for_shuffle])
    query = f""" SELECT {select_list_str} FROM {SNOWFLAKE_DATA_TABLE} WHERE stage_rank <= {rank_threshold}; """
    try:
        candidates_df = _session.sql(query).to_pandas()
        # Rename columns
        candidates_df.columns = candidates_df.columns.str.lower()
        candidates_df = candidates_df.rename(columns={"name": "song_name", "id": "song_id"})
        print(f"[{time.time()}] Loaded {len(candidates_df)} candidate rows from Snowflake.")
        return candidates_df
    except Exception as e: st.error(f"Error loading candidate songs from Snowflake: {e}"); return pd.DataFrame()

# --- The rest of the functions remain the same as they operate on Pandas DataFrames ---
# ... (get_shuffle_journey_pandas, calculate_single_song_scores, display_pagination_controls) ...
# ... (run_shuffle_generation, set_selected_track) ...
def get_shuffle_journey_pandas(candidates_df, stage_map):
    """Generates a 12-song journey using Pandas sampling on pre-loaded candidates."""
    print(f"[{time.time()}] --- Running get_shuffle_journey_pandas ---")
    if candidates_df.empty: print(f"[{time.time()}] Candidate DataFrame is empty."); return pd.DataFrame(), False
    journey_list = []; all_stages_found = True
    if 'stage_number' not in candidates_df.columns: print(f"[{time.time()}] Error: 'stage_number' column missing from candidates_df."); return pd.DataFrame(), False
    candidates_df['stage_number'] = candidates_df['stage_number'].astype(int)
    for stage_num in range(1, 13):
        stage_candidates = candidates_df[candidates_df['stage_number'] == stage_num]
        if not stage_candidates.empty:
            stage_candidates_sorted = stage_candidates.sort_values(by=['stage_rank', 'total_mismatch_score'])
            sampled_song = stage_candidates_sorted.sample(n=1, random_state=None).iloc[0]
            journey_list.append(sampled_song.to_dict())
        else:
            # *** Use song_name and song_id keys for consistency ***
            journey_list.append({'stage_number': stage_num, 'song_name': f'No Candidate Song Found for Stage {stage_num}', 'artists': '', 'album': '', 'track_uri': '', 'song_id': '', 'stage_rank': -1, 'total_mismatch_score': -1})
            all_stages_found = False; print(f"[{time.time()}] No candidate song found for stage {stage_num}")
    journey_df = pd.DataFrame(journey_list)
    journey_df['stage'] = journey_df['stage_number'].apply(lambda num: f"{num} - {stage_map.get(num, 'Unknown')}")
    journey_df['stage_number'] = journey_df['stage_number'].astype(int)
    print(f"[{time.time()}] Finished generating pandas journey. All stages found: {all_stages_found}")
    return journey_df.sort_values(by='stage_number').reset_index(drop=True), not all_stages_found

def calculate_single_song_scores(song_features, current_specs):
    """Calculates mismatch scores for one song against current specs."""
    # ... (previous implementation is fine) ...
    scores = []
    if not current_specs: return pd.DataFrame()
    first_stage_data = next(iter(current_specs.values()))
    features_in_spec = [k for k in first_stage_data if k != 'stage_number']
    for stage_name, spec_details in current_specs.items():
        stage_num = spec_details.get('stage_number', -1); total_distance = 0; features_calculated = 0
        for feature_name in features_in_spec:
            song_feature_key = next((k for k in song_features if k.lower() == feature_name.lower()), None)
            if feature_name not in spec_details or song_feature_key is None: continue
            spec_range = spec_details[feature_name]; song_value = song_features[song_feature_key]
            if pd.isna(song_value): continue
            if not isinstance(spec_range, list) or len(spec_range) != 2: continue
            try:
                min_val, max_val = float(spec_range[0]), float(spec_range[1]); song_val_float = float(song_value)
                range_center = (min_val + max_val) / 2.0; range_half_width = (max_val - min_val) / 2.0
                distance = 0
                if range_half_width <= 1e-9: distance = 0 if abs(song_val_float - range_center) < 1e-9 else 1000
                else: distance = abs(song_val_float - range_center) / range_half_width
                total_distance += distance; features_calculated += 1
            except (ValueError, TypeError, ZeroDivisionError) as e: continue
        if features_calculated > 0:
            scores.append({"stage_name": stage_name, "stage_number": stage_num, "total_mismatch_score": total_distance, "features_calculated": features_calculated})
    scores_df = pd.DataFrame(scores)
    if not scores_df.empty:
        scores_df = scores_df.sort_values(by="total_mismatch_score", ascending=True).reset_index(drop=True)
        scores_df['calculated_rank'] = scores_df.index + 1
    return scores_df

def display_pagination_controls(total_rows, key_prefix):
    """Displays pagination controls below the table and returns limit and offset."""
    # ... (previous implementation is fine) ...
    col1, col2, col3, col4, col5 = st.columns([2, 3, 1, 2, 1])
    rows_per_page_key = f"{key_prefix}_rows_per_page"; current_page_key = f"{key_prefix}_page"
    with col1:
        current_rows_per_page = st.session_state[rows_per_page_key]
        new_rows_per_page = st.selectbox("Rows per page:", options=ROWS_PER_PAGE_OPTIONS, index=ROWS_PER_PAGE_OPTIONS.index(current_rows_per_page), key=f"{key_prefix}_rows_per_page_widget")
        if new_rows_per_page != current_rows_per_page: st.session_state[rows_per_page_key] = new_rows_per_page; st.session_state[current_page_key] = 1; st.rerun()
    limit = st.session_state[rows_per_page_key]; total_pages = math.ceil(total_rows / limit) if limit > 0 else 1
    if current_page_key not in st.session_state: st.session_state[current_page_key] = 1
    if st.session_state[current_page_key] > total_pages: st.session_state[current_page_key] = max(1, total_pages)
    if st.session_state[current_page_key] < 1: st.session_state[current_page_key] = 1
    # Define callback functions for pagination buttons
    def prev_page(): st.session_state[current_page_key] -= 1
    def next_page(): st.session_state[current_page_key] += 1
    with col3: st.button("⬅️", key=f"{key_prefix}_prev", disabled=(st.session_state[current_page_key] <= 1), use_container_width=True, on_click=prev_page)
    with col4: st.write(f"Page **{st.session_state[current_page_key]}** of **{total_pages}**"); st.caption(f"(__{total_rows:,}__ rows)")
    with col5: st.button("➡️", key=f"{key_prefix}_next", disabled=(st.session_state[current_page_key] >= total_pages), use_container_width=True, on_click=next_page)
    offset = (st.session_state[current_page_key] - 1) * limit; return limit, offset

# --- Callback Function for Shuffle Button ---
def run_shuffle_generation():
    """Callback function to generate shuffle using Pandas and store in session state."""
    print(f"[{time.time()}] Shuffle button callback triggered (Pandas version).") # DEBUG
    if 'candidate_songs_df' in globals() and candidate_songs_df is not None and not candidate_songs_df.empty and stage_mapping:
        with st.spinner("Generating journey..."):
            st.session_state.shuffle_df, st.session_state.shuffle_fallback = get_shuffle_journey_pandas(candidate_songs_df, stage_mapping)
        print(f"[{time.time()}] Shuffle data stored in session state via callback (Pandas).") # DEBUG
        st.session_state.current_view = "Shuffle Journey" # Explicitly set view after generation
    else:
        st.session_state.shuffle_df = None; st.session_state.shuffle_fallback = False
        error_msg = "Cannot generate shuffle: ";
        if 'candidate_songs_df' not in globals() or candidate_songs_df is None or candidate_songs_df.empty: error_msg += "Candidate songs not loaded or empty. "
        if not stage_mapping: error_msg += "Missing stage mapping."
        st.toast(error_msg, icon="⚠️")
    # *** No st.rerun() here ***

# --- Callback function for setting selected track ---
def set_selected_track(uri):
    """Callback to update the selected track URI in session state."""
    print(f"[{time.time()}] Setting selected track URI: {uri}") # DEBUG
    st.session_state.selected_track_uri = uri
    # *** No st.rerun() here ***

# --- Streamlit App UI ---

# st.title("✨ Sound Journey Stage Explorer") # Removed title

# Check prerequisites
if not session: st.error("Could not establish Snowflake session. Stopping."); st.stop()
if not STAGE_SPECS: st.error(f"Error: Stage specifications not loaded correctly from '{CONFIG_FILE}'. Stopping."); st.stop()

# --- Fetch initial data for UI elements ---
with st.spinner("Loading initial data and candidate songs..."):
    stage_mapping = get_stage_mapping()
    # Use Snowflake query functions
    feature_ranges = get_feature_ranges_snowflake(session, CONTINUOUS_FEATURES)
    discrete_values = get_discrete_feature_values_snowflake(session, DISCRETE_FEATURES)
    candidate_songs_df = load_candidate_songs_snowflake(session, rank_threshold=1)

if not stage_mapping: st.warning("Could not load stage information from config.")

# --- Sidebar ---
with st.sidebar:
    st.header("Navigation")
    # Use radio buttons for view selection instead of tabs
    query_params = st.query_params.to_dict()
    default_view = VIEWS[0]
    current_view_from_state = st.session_state.get('current_view', default_view)
    if current_view_from_state not in VIEWS: current_view_from_state = default_view
    query_param_view = query_params.get("view", [current_view_from_state])[0]
    if query_param_view not in VIEWS: query_param_view = current_view_from_state
    try: initial_view_index = VIEWS.index(query_param_view)
    except ValueError: initial_view_index = 0

    def update_view():
        st.session_state.current_view = st.session_state.view_selector
        try: st.query_params["view"] = st.session_state.view_selector
        except AttributeError: print("Warning: st.query_params not available.")

    st.radio(
        "Select View:", options=VIEWS, index=initial_view_index,
        key="view_selector", on_change=update_view,
        format_func=lambda x: f"{VIEW_ICONS[VIEWS.index(x)]} {x}",
        horizontal=True
    )
    st.divider()

    # --- Player Area (Moved to Sidebar) ---
    player_placeholder = st.container()
    with player_placeholder:
        st.subheader("🎧 Player")
        if 'selected_track_uri' in st.session_state and st.session_state.selected_track_uri:
            selected_uri = st.session_state.selected_track_uri
            track_id_match = re.search(r'spotify:track:(\w+)', selected_uri)
            if track_id_match:
                track_id = track_id_match.group(1)
                embed_url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator"
                iframe_html = f'''<iframe style="border-radius:12px;" src="{embed_url}" width="100%" height="80" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>'''
                st.components.v1.html(iframe_html, height=80)
                st.button("Clear Player ⏹️", key="clear_player", on_click=set_selected_track, args=(None,))
            else:
                if selected_uri is not None: st.warning(f"Invalid Track URI.")
                if st.session_state.selected_track_uri == selected_uri: st.session_state.selected_track_uri = None;
        else:
            st.caption("Click '▶️ Play' on a song row to load player.")
        st.markdown("---")

    st.header("⚙️ Filters & Options")
    active_filters = {}
    st.subheader("General Filters")
    active_filters['song'] = st.text_input("Filter by Song Name (contains):")
    active_filters['artist'] = st.text_input("Filter by Artist Name (contains):")
    st.divider()

    st.subheader("Audio Features")
    audio_features_for_filtering = [f for f in FEATURES_FOR_UI if f not in ['stage_name', 'stage_number', 'stage_rank']]
    for feature in audio_features_for_filtering:
        feature_def = definitions.get(feature.lower(), f"Filter by {feature.replace('_',' ')}")
        if feature in CONTINUOUS_FEATURES:
            min_val, max_val = feature_ranges.get(feature, (0.0, 1.0))
            if feature == 'total_mismatch_score': min_val, max_val = feature_ranges.get(feature, (0.0, DEFAULT_MAX_SCORE))
            if min_val >= max_val:
                 if min_val == max_val: st.text(f"{feature.capitalize()}: {min_val}")
                 else: st.text(f"{feature.capitalize()}: Invalid range")
                 active_filters[feature] = None
            else:
                 f_min, f_max = float(min_val), float(max_val)
                 step = 0.5 if feature == 'total_mismatch_score' else ((f_max - f_min) / 100 if (f_max - f_min) > 0 else 0.01)
                 default_value = f_max if feature == 'total_mismatch_score' else (f_min, f_max)
                 if feature == 'total_mismatch_score':
                      active_filters[feature] = st.slider(f"Max Mismatch Score:", min_value=f_min, max_value=f_max, value=default_value, step=step, key=f"slider_{feature}", help=feature_def)
                 else:
                      active_filters[feature] = st.slider(f"{feature.replace('_',' ').capitalize()} Range:", min_value=f_min, max_value=f_max, value=default_value, step=step, key=f"slider_{feature}", help=feature_def)
        elif feature in DISCRETE_FEATURES:
            options = discrete_values.get(feature, []);
            if options: active_filters[feature] = st.multiselect(f"{feature.replace('_',' ').capitalize()} Select:", options=options, default=[], key=f"multi_{feature}", help=feature_def)
            else: active_filters[feature] = []

# --- Main Content Area (Controlled by Sidebar Radio) ---
current_view = st.session_state.current_view

if current_view == "Stage Explorer":
    st.subheader("Explore Songs by Stage")
    stage_display_options = {num: f"{num} - {name}" for num, name in stage_mapping.items()}
    selected_stage_display = st.selectbox("Select a Stage:", options=list(stage_display_options.values()), index=0, key="explorer_stage_select")
    selected_stage_number = None
    for num, display_val in stage_display_options.items():
         if display_val == selected_stage_display: selected_stage_number = num; break
    if selected_stage_number is not None:
        total_rows = count_songs_snowflake(session, selected_stage_number, tuple(sorted(active_filters.items())), feature_ranges)
        limit, offset = st.session_state.explorer_rows_per_page, (st.session_state.explorer_page - 1) * st.session_state.explorer_rows_per_page
        results_df = query_paged_songs_snowflake(session, selected_stage_number, active_filters, limit, offset, feature_ranges)

        if not results_df.empty:
            results_df_display = results_df.copy()
            results_df_display.insert(0, "▶️ Play", False)
            display_cols_explorer = ["▶️ Play", 'stage_rank', 'song_name', 'artists', 'album', 'total_mismatch_score'] + ALL_AUDIO_FEATURES
            final_display_cols = [col for col in display_cols_explorer if col in results_df_display.columns]
            column_config = { "▶️ Play": st.column_config.CheckboxColumn("Play", default=False, width="small") }
            for feat in ALL_AUDIO_FEATURES:
                 if feat == 'loudness': column_config[feat] = st.column_config.NumberColumn(format="%.2f dB")
                 elif feat == 'tempo': column_config[feat] = st.column_config.NumberColumn(format="%.1f BPM")
                 elif feat == 'total_mismatch_score': column_config[feat] = st.column_config.NumberColumn(format="%.2f")
                 elif feat in CONTINUOUS_FEATURES: column_config[feat] = st.column_config.NumberColumn(format="%.3f")
                 if feat not in final_display_cols: column_config[feat] = None
            column_config.update({"track_uri": None, "song_id": None, "stage_name": None, "stage_number": None})

            edited_df = st.data_editor(
                results_df_display[final_display_cols], use_container_width=True, hide_index=True,
                column_config=column_config, key="explorer_editor",
                disabled=list(results_df_display.columns.drop("▶️ Play"))
            )
            try:
                 clicked_rows = edited_df[edited_df["▶️ Play"] == True]
                 if not clicked_rows.empty:
                      clicked_row_index = clicked_rows.index[0]
                      selected_row_original = results_df.iloc[clicked_row_index]
                      selected_uri = selected_row_original.get('track_uri')
                      if selected_uri and st.session_state.selected_track_uri != selected_uri:
                           set_selected_track(selected_uri); st.rerun()
            except IndexError: print(f"IndexError during selection processing for explorer.")
            except Exception as e: st.error(f"Error processing table selection: {e}")

            display_pagination_controls(total_rows, key_prefix="explorer")
        else: st.info(f"No songs found matching the current filters for Stage {selected_stage_number}.")
    else: st.warning("Could not determine selected stage number.")

elif current_view == "All Catalog":
    st.subheader("Browse All Songs")
    total_rows_catalog = count_songs_snowflake(session, None, tuple(sorted(active_filters.items())), feature_ranges)
    limit_catalog, offset_catalog = st.session_state.catalog_rows_per_page, (st.session_state.catalog_page - 1) * st.session_state.catalog_rows_per_page
    results_df_catalog = query_paged_songs_snowflake(session, None, active_filters, limit_catalog, offset_catalog, feature_ranges)

    if not results_df_catalog.empty:
        results_df_catalog_display = results_df_catalog.copy()
        results_df_catalog_display.insert(0, "▶️ Play", False)
        display_cols_catalog = ["▶️ Play", 'song_name', 'artists', 'album', 'stage_number', 'stage_name', 'stage_rank', 'total_mismatch_score'] + ALL_AUDIO_FEATURES
        final_display_cols_cat = [col for col in display_cols_catalog if col in results_df_catalog_display.columns]
        column_config_catalog = { "▶️ Play": st.column_config.CheckboxColumn("Play", default=False, width="small") }
        for feat in ALL_AUDIO_FEATURES:
             if feat == 'loudness': column_config_catalog[feat] = st.column_config.NumberColumn(format="%.2f dB")
             elif feat == 'tempo': column_config_catalog[feat] = st.column_config.NumberColumn(format="%.1f BPM")
             elif feat == 'total_mismatch_score': column_config_catalog[feat] = st.column_config.NumberColumn(format="%.2f")
             elif feat in CONTINUOUS_FEATURES: column_config_catalog[feat] = st.column_config.NumberColumn(format="%.3f")
             if feat not in final_display_cols_cat: column_config_catalog[feat] = None
        column_config_catalog.update({"track_uri": None, "song_id": None})

        edited_df_catalog = st.data_editor(
            results_df_catalog_display[final_display_cols_cat], use_container_width=True, hide_index=True,
            column_config=column_config_catalog, key="catalog_editor",
            disabled=list(results_df_catalog_display.columns.drop("▶️ Play"))
        )
        try:
            diff_catalog = edited_df_catalog[edited_df_catalog["▶️ Play"] == True]
            if not diff_catalog.empty:
                clicked_row_index_cat = diff_catalog.index[0]
                selected_row_original_cat = results_df_catalog.iloc[clicked_row_index_cat]
                selected_uri_cat = selected_row_original_cat.get('track_uri')
                if selected_uri_cat and st.session_state.selected_track_uri != selected_uri_cat:
                    set_selected_track(selected_uri_cat); st.rerun()
        except IndexError: print(f"IndexError during selection processing for catalog.")
        except Exception as e: st.error(f"Error processing table selection: {e}")

        display_pagination_controls(total_rows_catalog, key_prefix="catalog")
    else: st.info("No songs found matching the current filters.")

elif current_view == "Shuffle Journey":
    st.subheader("Generate a Random Sound Journey")
    st.button("🔀 Generate Shuffle Journey", key="generate_shuffle_button_pandas", on_click=run_shuffle_generation)
    shuffle_results_area = st.container()
    with shuffle_results_area:
        if 'shuffle_df' in st.session_state and st.session_state.shuffle_df is not None:
            journey_df_to_display = st.session_state.shuffle_df
            fallback_used_to_display = st.session_state.shuffle_fallback
            if not journey_df_to_display.empty:
                if fallback_used_to_display: st.caption("ℹ️ _Note: Some stages lacked Rank 1 songs; best available used._")
                journey_df_display_editor = journey_df_to_display.copy()
                journey_df_display_editor.insert(0, "▶️ Play", False)
                display_cols_shuffle = ["▶️ Play", 'stage', 'song_name', 'artists', 'album', 'stage_rank', 'total_mismatch_score']
                actual_cols_shuffle = [col for col in display_cols_shuffle if col in journey_df_display_editor.columns]
                column_config_shuffle = {
                    "▶️ Play": st.column_config.CheckboxColumn("Play", default=False, width="small"),
                    "track_uri": None, "song_id": None,
                    "total_mismatch_score": st.column_config.NumberColumn(format="%.2f"),
                }
                edited_df_shuffle = st.data_editor(
                    journey_df_display_editor[actual_cols_shuffle], use_container_width=True, hide_index=True,
                    column_config=column_config_shuffle, key="shuffle_editor",
                    disabled=list(journey_df_display_editor.columns.drop("▶️ Play"))
                )
                try:
                    diff_shuffle = edited_df_shuffle[edited_df_shuffle["▶️ Play"] == True]
                    if not diff_shuffle.empty:
                        clicked_row_index_shuf = diff_shuffle.index[0]
                        selected_row_original_shuf = journey_df_to_display.iloc[clicked_row_index_shuf]
                        selected_uri_shuf = selected_row_original_shuf.get('track_uri')
                        if selected_uri_shuf and st.session_state.selected_track_uri != selected_uri_shuf:
                            set_selected_track(selected_uri_shuf); st.rerun()
                except IndexError: print(f"IndexError during selection processing for shuffle.")
                except Exception as e: st.error(f"Error processing table selection: {e}")
            else: st.warning("Could not generate journey or no results found.")

elif current_view == "Spec Tester":
    st.subheader("Test Song Classification with Current Specs")
    st.write(f"These calculations use the specifications currently loaded from `{CONFIG_FILE}`.")
    st.info(f"Edit the `config.yaml` file and **reload this page (F5 or Cmd+R)** in your browser to test different specs.")
    with st.expander("View Current Stage Specifications from config.yaml"): st.json(STAGE_SPECS)
    st.divider()
    st.session_state.test_song_id = st.text_input("Enter Song ID or partial name to test:", value=st.session_state.test_song_id, key="test_song_input", placeholder="e.g., 4iV5W9uYEdYUVa79Axb7Rh or Bohemian Rhapsody")
    if st.session_state.test_song_id:
        search_term = st.session_state.test_song_id.replace("'", "''")
        # Query Snowflake table for the specific song ID or name
        find_query = f"""
            SELECT * FROM {SNOWFLAKE_DATA_TABLE}
            WHERE id = :search_id OR lower(name) LIKE lower(:search_name)
            LIMIT 50;
        """
        params = {"search_id": search_term, "search_name": f"%{search_term}%"}
        try:
            possible_songs_df = session.sql(find_query, params=params).to_pandas()
            # Convert column names to lowercase for consistency
            possible_songs_df.columns = possible_songs_df.columns.str.lower()
            # Rename id and name to match expected Python names
            possible_songs_df = possible_songs_df.rename(columns={"name": "song_name", "id": "song_id"})


            if not possible_songs_df.empty:
                if len(possible_songs_df) > 1:
                     possible_songs_df['display_option'] = possible_songs_df.apply(lambda row: f"{row['song_name']} by {row.get('artists','Unknown')} (ID: {row['song_id']})", axis=1)
                     selected_song_display = st.selectbox("Select the exact song:", options=possible_songs_df['display_option'].tolist(), index=0)
                     try: selected_song_id = selected_song_display.split("(ID: ")[1].replace(")", "")
                     except IndexError: st.error("Could not parse selected song ID."); st.stop()
                     song_features_series = possible_songs_df[possible_songs_df['song_id'] == selected_song_id].iloc[0]
                else:
                     song_features_series = possible_songs_df.iloc[0]; selected_song_id = song_features_series['song_id']
                     st.write(f"Found song: **{song_features_series['song_name']}** by {song_features_series.get('artists','Unknown')}")

                # Extract features needed for calculation
                song_features_dict = {}
                for feature in ALL_FEATURES_FROM_SPEC:
                     if feature.lower() in song_features_series.index:
                          song_features_dict[feature] = song_features_series[feature.lower()]
                     else:
                          print(f"Warning: Feature '{feature}' from spec not found in song data for testing."); song_features_dict[feature] = None

                st.write("Calculating scores based on current specs...")
                calculated_scores_df = calculate_single_song_scores(song_features_dict, STAGE_SPECS)
                if not calculated_scores_df.empty:
                    st.write(f"Calculated Stage Ranks for Song ID: {selected_song_id}")
                    calculated_scores_df['stage'] = calculated_scores_df['stage_number'].apply(lambda num: f"{stage_mapping.get(num, 'Unknown')}")
                    display_cols_test = ['calculated_rank', 'stage', 'total_mismatch_score', 'features_calculated']
                    st.dataframe(calculated_scores_df[display_cols_test], hide_index=True)
                    with st.expander("View Song's Audio Features Used for Calculation"): st.json({k: v for k, v in song_features_dict.items() if pd.notna(v) and k in ALL_FEATURES_FROM_SPEC})
                else: st.warning("Could not calculate scores for this song with the current specs.")
            else: st.warning(f"No song found matching ID or name '{st.session_state.test_song_id}'.")
        except Exception as e: st.error(f"Error finding or testing song: {e}"); st.error(f"Query attempted: {find_query} with params: {params}")

