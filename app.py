# -*- coding: utf-8 -*-
"""
Streamlit app to explore and shuffle songs classified by stage,
querying data directly from GCS using DuckDB.
Reads configuration AND stage specs from config.yaml file.
Includes comprehensive filtering, pagination below tables, an all catalog view,
improved shuffle logic, UI enhancements, a Spec Tester tab,
and an embedded Spotify player loaded using st.data_editor interaction.
Uses sidebar radio for navigation. Reads GCS credentials from st.secrets
and uses a temporary key file for DuckDB authentication.
Added basic password protection.
"""

import streamlit as st
import duckdb
import pandas as pd
import os
import random
import math
import yaml # Import YAML library
import time # For debugging timestamp
import csv # Import csv library again
import re # For extracting track ID
import json # For parsing service account key
import tempfile # To create temporary file for GCS key

# --- Page Configuration (MUST be the first Streamlit command after imports) ---
st.set_page_config(layout="wide", page_title="Sound Journey Explorer")

# --- Password Protection ---
def check_password():
    """Returns `True` if the user had the correct password."""
    correct_password = st.secrets.get("PASSWORD")
    if not correct_password:
        is_deployed = "STREAMLIT_SERVER_RUNNING_MODE" in os.environ
        if is_deployed: st.error("Password not configured."); st.stop(); return False
        else: st.warning("Password secret not found."); return True # Bypass locally
    def password_entered():
        if st.session_state["password"] == correct_password: st.session_state["password_correct"] = True; del st.session_state["password"]
        else: st.session_state["password_correct"] = False
    if st.session_state.get("password_correct", False): return True
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]: st.error("😕 Password incorrect")
    elif "password_correct" not in st.session_state: st.info("Please enter the password to access the app.")
    return False

if not check_password(): st.stop()

# --- Load Configuration ---
CONFIG_FILE = 'config.yaml'
DEFINITIONS_FILE = 'Spotify_Data_Dictionary.csv'

@st.cache_data(ttl=30)
def load_config():
    """Loads configuration and stage specs from the YAML file."""
    if not os.path.exists(CONFIG_FILE): return None, f"Config file '{CONFIG_FILE}' not found."
    try:
        with open(CONFIG_FILE, 'r') as f: config_data = yaml.safe_load(f)
        if not config_data or 'local_data_path' not in config_data or 'file_format' not in config_data or 'stage_specs' not in config_data:
             return None, f"Config file '{CONFIG_FILE}' missing required keys ('local_data_path', 'file_format', 'stage_specs')."
        gcs_path = config_data.get('local_data_path')
        is_deployed = "STREAMLIT_SERVER_RUNNING_MODE" in os.environ
        if not gcs_path:
             return None, "Missing 'local_data_path' in config file."
        # If deployed, path MUST be GCS
        if is_deployed and not gcs_path.startswith('gs://'):
             return None, f"Invalid GCS path in config.yaml: '{gcs_path}'. Must start with 'gs://' for deployment."
        # If local, check if it's GCS (warn) or local (check existence)
        elif not is_deployed:
            if gcs_path.startswith('gs://'):
                 print(f"Warning: GCS path specified in config.yaml while running locally. Ensure GCS access is configured.")
            elif not os.path.exists(gcs_path):
                 return None, f"Local data file specified in config.yaml not found at: '{gcs_path}'"
        return config_data, None
    except yaml.YAMLError as e: return None, f"Error parsing config file '{CONFIG_FILE}': {e}"
    except Exception as e: return None, f"Error reading config file '{CONFIG_FILE}': {e}"

app_config, config_error_msg = load_config()

@st.cache_data(ttl=3600)
def load_definitions(filepath):
    """Loads feature definitions from the CSV file, trying different encodings."""
    definitions = {}
    if not os.path.exists(filepath): st.warning(f"Definitions file '{filepath}' not found."); return definitions
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

DATA_PATH = app_config.get('local_data_path') # Holds GCS or Local path
FILE_FORMAT = app_config.get('file_format', 'parquet').lower()
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


# --- DuckDB Setup ---
@st.cache_resource
def get_duckdb_connection():
    """Establishes a connection to DuckDB and configures GCS access using secrets via temp file."""
    is_deployed = "STREAMLIT_SERVER_RUNNING_MODE" in os.environ
    print(f"[{time.time()}] Attempting DuckDB connection. Deployed: {is_deployed}")

    # Check for GCS credentials in secrets if needed (path starts with gs://)
    gcs_key_json_content = None
    temp_key_file_path = None
    if DATA_PATH.startswith("gs://"):
        gcs_key_json_content = st.secrets.get("GCS_SERVICE_ACCOUNT_KEY_JSON")
        if not gcs_key_json_content or not gcs_key_json_content.strip():
            st.error("GCS Service Account Key not found or empty in Streamlit secrets.")
            return None
        try:
            # Validate JSON structure early
            json.loads(gcs_key_json_content)
            # Create a temporary file to store the key
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_key_file:
                temp_key_file.write(gcs_key_json_content)
                temp_key_file_path = temp_key_file.name # Get the path
            print(f"[{time.time()}] GCS key written to temporary file: {temp_key_file_path}")
        except json.JSONDecodeError:
            st.error("GCS Service Account Key in secrets is not valid JSON.")
            return None
        except Exception as e:
             st.error(f"Error writing GCS key to temporary file: {e}")
             return None

    try:
        con = duckdb.connect(database=':memory:', read_only=False)
        print(f"[{time.time()}] DuckDB connection established.")

        # Install and load GCS extension only if needed
        if DATA_PATH.startswith("gs://"):
            try:
                con.sql("INSTALL gcs;")
                con.sql("LOAD gcs;")
                print(f"[{time.time()}] DuckDB GCS extension loaded.")

                # Configure GCS credentials using the temporary key file path
                if temp_key_file_path:
                    # Escape backslashes for Windows paths if running locally, though unlikely needed for tempfile path
                    safe_key_path = temp_key_file_path.replace('\\', '\\\\')
                    # Use DuckDB SET commands to point to the key file
                    con.sql(f"SET gcs_service_account_key_file='{safe_key_path}';")
                    print(f"[{time.time()}] Configured DuckDB GCS credentials using key file: {safe_key_path}")
                else:
                    # This case should not happen if GCS path is used and key is required
                    st.warning("GCS path detected but failed to create temporary key file. Trying default credentials.")

            except Exception as e:
                 st.error(f"Failed to install/load DuckDB GCS extension or set credentials: {e}")
                 con.close()
                 # Clean up temp file if it exists and connection failed
                 if temp_key_file_path and os.path.exists(temp_key_file_path):
                     try: os.remove(temp_key_file_path)
                     except: pass
                 return None

        return con

    except Exception as e:
        st.error(f"Error connecting to DuckDB: {e}")
        if 'con' in locals() and con:
            try: con.close()
            except: pass
        # Clean up temp file if it exists and connection failed
        if temp_key_file_path and os.path.exists(temp_key_file_path):
            try: os.remove(temp_key_file_path)
            except: pass
        return None

con = get_duckdb_connection()

# --- Data Loading and Querying Functions ---

def get_read_options():
    """Generates DuckDB read options based on file format."""
    read_options = ""
    if FILE_FORMAT == 'csv':
        read_options = ", header=true"
        if DATA_PATH.endswith('.gz'): # Check path for compression
            read_options += ", compression='gzip'"
    return read_options

def read_from_source():
    """Helper function to read from the GCS/Local file source defined in config."""
    read_function = f"read_{FILE_FORMAT}"
    read_options = get_read_options()
    path = app_config['local_data_path'] # This holds GCS or Local path
    # *** REMOVED secret_option as credentials are set globally via SET command ***
    # Escape single quotes in path just in case
    safe_path = path.replace("'", "''")
    return f"{read_function}('{safe_path}'{read_options})"

# --- The rest of the functions (get_stage_mapping, get_feature_ranges, etc.)
# --- remain largely the same, as they use read_from_source() which now handles GCS/Local ---
# --- Re-paste all function definitions here from the previous version ---
# ... (get_stage_mapping, get_feature_ranges, get_discrete_feature_values) ...
# ... (build_where_clause, count_songs, query_paged_songs) ...
# ... (load_candidate_songs, get_shuffle_journey_pandas) ...
# ... (calculate_single_song_scores, display_pagination_controls) ...
# ... (run_shuffle_generation, set_selected_track) ...
def get_stage_mapping():
    mapping = {}
    for name, details in STAGE_SPECS.items():
        num = details.get('stage_number')
        if num is not None: mapping[num] = name
        else: print(f"Warning: Missing 'stage_number' for stage '{name}' in config.")
    return dict(sorted(mapping.items()))

@st.cache_data(ttl=3600)
def get_feature_ranges(_con, features):
    if not _con: return {}
    ranges = {}
    try:
        # Use DESCRIBE only if it's likely a local file for performance
        if not DATA_PATH.startswith("gs://"):
             temp_df = _con.query(f"DESCRIBE SELECT * FROM {read_from_source()}").df(); available_columns = temp_df['column_name'].str.lower().tolist()
        else: # Assume all features exist for remote files to avoid slow DESCRIBE over GCS
             available_columns = [f.lower() for f in features]
    except Exception as e:
        st.warning(f"Could not describe table to get feature ranges: {e}")
        available_columns = [f.lower() for f in features] # Assume all exist on error

    features_to_query = [f for f in features if f.lower() in available_columns]
    if not features_to_query: st.warning("Could not find expected feature columns in data file to determine ranges."); return {feat: (0.0, 100.0) if feat == 'total_mismatch_score' else (0.0, 1.0) for feat in features}
    valid_features = [f for f in features_to_query if f != 'total_mismatch_score']
    select_clauses = [f"MIN(\"{feat}\") as min_{feat}, MAX(\"{feat}\") as max_{feat}" for feat in valid_features] # Quote feature names
    if 'total_mismatch_score' in features_to_query: select_clauses.append("MIN(total_mismatch_score) as min_total_mismatch_score, MAX(total_mismatch_score) as max_total_mismatch_score")
    if not select_clauses: return {feat: (0.0, 100.0) if feat == 'total_mismatch_score' else (0.0, 1.0) for feat in features}
    query = f"SELECT {', '.join(select_clauses)} FROM {read_from_source()}"
    try:
        result = _con.query(query).df()
        if not result.empty:
            row = result.iloc[0]
            for feat in features:
                min_val = row.get(f'min_{feat}', None); max_val = row.get(f'max_{feat}', None)
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
        st.warning(f"Could not fetch feature ranges: {e}")
        return {feat: (0.0, DEFAULT_MAX_SCORE) if feat == 'total_mismatch_score' else (0.0, 1.0) for feat in features}

@st.cache_data(ttl=3600)
def get_discrete_feature_values(_con, features):
    if not _con: return {}
    values = {}
    try:
        # Avoid DESCRIBE over GCS if possible
        if not DATA_PATH.startswith("gs://"):
            temp_df = _con.query(f"DESCRIBE SELECT * FROM {read_from_source()}").df(); available_columns = temp_df['column_name'].str.lower().tolist()
        else:
             available_columns = [f.lower() for f in features] # Assume columns exist
    except Exception as e:
        st.warning(f"Could not describe discrete feature columns: {e}")
        available_columns = [f.lower() for f in features] # Assume columns exist
    try:
        for feat in features:
            if feat.lower() not in available_columns: print(f"Warning: Discrete feature '{feat}' not found in data file."); values[feat] = []; continue
            # Quote column name in query
            query = f"SELECT DISTINCT \"{feat}\" FROM {read_from_source()} WHERE \"{feat}\" IS NOT NULL ORDER BY \"{feat}\""
            result = _con.query(query).df()
            try: values[feat] = sorted(result[feat].astype(int).tolist()) if not result.empty else []
            except (ValueError, TypeError): values[feat] = sorted(result[feat].tolist()) if not result.empty else []
        return values
    except Exception as e:
        st.warning(f"Could not fetch discrete feature values for {feat}: {e}"); return {feat: [] for feat in features}

def build_where_clause(stage_num, filters, feature_ranges_local):
    where_clauses = []
    if stage_num is not None: where_clauses.append(f"stage_number = {stage_num}")
    # Add score filter only if viewing a specific stage
    if stage_num is not None and 'score' in filters and filters['score'] is not None:
        score_max_val = feature_ranges_local.get('total_mismatch_score', (0.0, DEFAULT_MAX_SCORE))[1]
        effective_max_score = min(filters['score'], score_max_val, 1000.0)
        where_clauses.append(f"total_mismatch_score <= {effective_max_score}") # No quotes needed for standard name

    # Removed general filters
    # if 'artist' in filters and filters['artist']: ...
    # if 'song' in filters and filters['song']: ...

    for feat in CONTINUOUS_FEATURES:
        # Skip score filter here
        if feat == 'total_mismatch_score': continue
        if feat in filters and filters[feat] is not None:
            min_val, max_val = filters[feat]
            default_min, default_max = feature_ranges_local.get(feat, (0,1))
            if min_val is not None and max_val is not None and (min_val > default_min or max_val < default_max):
                 where_clauses.append(f"\"{feat}\" >= {min_val} AND \"{feat}\" <= {max_val}") # Keep quotes for safety
    for feat in DISCRETE_FEATURES: # Now only 'key'
        if feat in filters and filters[feat]:
            quoted_values = []
            for v in filters[feat]:
                if isinstance(v, str): escaped_v = v.replace("'", "''"); quoted_values.append("'" + escaped_v + "'")
                else: quoted_values.append(str(v))
            selected_values = ", ".join(quoted_values)
            if selected_values: where_clauses.append(f"\"{feat}\" IN ({selected_values})") # Keep quotes for safety
    return "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

@st.cache_data(ttl=600)
def count_songs(_con, stage_num, filters_tuple, feature_ranges_local):
    filters = dict(filters_tuple)
    if not _con: return 0
    full_where_clause = build_where_clause(stage_num, filters, feature_ranges_local)
    # *** Use id for counting (assuming 'id' is the correct identifier column name) ***
    query = f"SELECT COUNT(DISTINCT id) as total_count FROM {read_from_source()} {full_where_clause}"
    try:
        result = _con.query(query).df(); return result['total_count'][0] if not result.empty else 0
    except Exception as e:
        st.error(f"Error counting songs: {e}"); st.error(f"Query attempted: {query}"); return 0

def query_paged_songs(_con, stage_num, filters, limit, offset, feature_ranges_local):
    if not _con: return pd.DataFrame()
    full_where_clause = build_where_clause(stage_num, filters, feature_ranges_local)
    try:
        # Avoid DESCRIBE over GCS if possible
        if not DATA_PATH.startswith("gs://"):
             temp_df = _con.query(f"DESCRIBE SELECT * FROM {read_from_source()}").df(); available_columns = temp_df['column_name'].tolist()
        else: # Assume columns exist based on lists
             available_columns = ['id', 'name', 'artists', 'album', 'track_uri', 'stage_number', 'stage_name', 'stage_rank', 'total_mismatch_score'] + ALL_AUDIO_FEATURES
    except Exception: st.error("Could not describe table columns. Query might fail."); available_columns = []
    # *** Use id and name in base columns ***
    base_select_columns = ['id', 'name', 'artists', 'album', 'track_uri', 'stage_number', 'stage_name', 'stage_rank', 'total_mismatch_score']
    available_ui_features = [f for f in ALL_AUDIO_FEATURES if f in available_columns]
    select_columns = base_select_columns + available_ui_features
    final_select_list = []; seen = set()
    for col in select_columns:
        actual_col_name = next((c for c in available_columns if c.lower() == col.lower()), None)
        if actual_col_name and actual_col_name.lower() not in seen:
            final_select_list.append(f'"{actual_col_name}"'); seen.add(actual_col_name.lower())
    if not final_select_list: st.error("No valid columns found to select from the data file."); return pd.DataFrame()
    # *** Use id in ORDER BY ***
    query = f"""
        SELECT {', '.join(final_select_list)} FROM {read_from_source()} {full_where_clause}
        ORDER BY CASE WHEN stage_number IS NOT NULL THEN stage_rank END ASC, total_mismatch_score ASC, id ASC
        LIMIT {limit} OFFSET {offset};
    """
    try:
        results_df = _con.query(query).df();
        # *** Rename id and name AFTER query ***
        results_df = results_df.rename(columns={"name": "song_name", "id": "song_id"});
        return results_df
    except Exception as e:
        st.error(f"Error querying paged data: {e}"); st.error(f"Query attempted: {query}"); return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_candidate_songs(_con, rank_threshold=1):
    """Loads potential candidate songs (e.g., rank 1) into a Pandas DataFrame."""
    if not _con: return pd.DataFrame()
    print(f"[{time.time()}] Loading candidate songs (Rank <= {rank_threshold})...")
    # *** Use id and name here ***
    cols_for_shuffle = ['id', 'name', 'artists', 'album', 'track_uri', 'stage_number', 'stage_name', 'stage_rank', 'total_mismatch_score']
    try:
        # Avoid DESCRIBE over GCS if possible
        if not DATA_PATH.startswith("gs://"):
             temp_df = _con.query(f"DESCRIBE SELECT * FROM {read_from_source()}").df(); available_columns = temp_df['column_name'].tolist()
        else: # Assume columns exist
             available_columns = cols_for_shuffle + ['key','mode','time_signature','danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo']

        # Select using actual case found in file, but check against lowercase list
        select_cols_shuffle = []
        cols_for_shuffle_lower = [c.lower() for c in cols_for_shuffle]
        for actual_col in available_columns:
             if actual_col.lower() in cols_for_shuffle_lower:
                  select_cols_shuffle.append(f'"{actual_col}"')

        if not select_cols_shuffle: st.error("Required columns for shuffle not found in data."); return pd.DataFrame()
    except Exception as e: st.error(f"Error describing table for shuffle candidates: {e}"); return pd.DataFrame()
    query = f""" SELECT {', '.join(select_cols_shuffle)} FROM {read_from_source()} WHERE stage_rank <= {rank_threshold}; """
    try:
        candidates_df = _con.query(query).df()
        # *** Rename AFTER loading ***
        candidates_df = candidates_df.rename(columns={"name": "song_name", "id": "song_id"})
        print(f"[{time.time()}] Loaded {len(candidates_df)} candidate rows.")
        return candidates_df
    except Exception as e: st.error(f"Error loading candidate songs: {e}"); return pd.DataFrame()

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
if not con: st.error("Could not establish DuckDB connection. Stopping."); st.stop()
# Check DATA_PATH
if not DATA_PATH: st.error("Data path not defined in config.yaml."); st.stop()
# Check if it's a GCS path or a local file that exists
if not DATA_PATH.startswith("gs://") and not os.path.exists(DATA_PATH):
     st.error(f"Error: Local data file not found at '{DATA_PATH}'. Check config.yaml.")
     st.stop()
elif DATA_PATH.startswith("gs://"):
     print(f"Attempting to read from GCS path: {DATA_PATH}") # Log GCS path usage

if not STAGE_SPECS: st.error(f"Error: Stage specifications not loaded correctly from '{CONFIG_FILE}'. Stopping."); st.stop()

# --- Fetch initial data for UI elements ---
with st.spinner("Loading initial data and candidate songs..."):
    stage_mapping = get_stage_mapping()
    feature_ranges = get_feature_ranges(con, CONTINUOUS_FEATURES)
    discrete_values = get_discrete_feature_values(con, DISCRETE_FEATURES) # Only load discrete features used for filtering
    # Load candidate songs for shuffle (using default rank 1)
    candidate_songs_df = load_candidate_songs(con, rank_threshold=1) # Load Rank 1 songs

if not stage_mapping: st.warning("Could not load stage information from config.")

# --- Sidebar ---
with st.sidebar:
    st.header("Navigation")
    # Use radio buttons for view selection instead of tabs
    query_params = st.query_params.to_dict()
    default_view = VIEWS[0]
    # Get current view from session state if it exists, otherwise use query param or default
    current_view_from_state = st.session_state.get('current_view', default_view)
    if current_view_from_state not in VIEWS:
        current_view_from_state = default_view # Fallback if state has invalid view

    query_param_view = query_params.get("view", [current_view_from_state])[0]
    if query_param_view not in VIEWS:
        query_param_view = current_view_from_state # Fallback to state if query param invalid
    try:
        initial_view_index = VIEWS.index(query_param_view)
    except ValueError:
        initial_view_index = 0 # Default to first view if name is somehow invalid

    # Update session state based on radio button selection
    def update_view():
        st.session_state.current_view = st.session_state.view_selector
        # Update URL query param - use set() which replaces existing
        # Use st.query_params.set() for newer Streamlit versions if available
        try:
            st.query_params["view"] = st.session_state.view_selector
        except AttributeError: # Fallback for older versions
            print("Warning: st.query_params not available in this Streamlit version.")


    st.radio(
        "Select View:", options=VIEWS, index=initial_view_index,
        key="view_selector", on_change=update_view,
        format_func=lambda x: f"{VIEW_ICONS[VIEWS.index(x)]} {x}",
        horizontal=True # Make radio horizontal
    )
    st.divider()

    # --- Player Area (Moved to Sidebar) ---
    player_placeholder = st.container() # Use a container within the sidebar column
    with player_placeholder:
        st.subheader("🎧 Player") # Add a subheader
        if 'selected_track_uri' in st.session_state and st.session_state.selected_track_uri:
            selected_uri = st.session_state.selected_track_uri
            track_id_match = re.search(r'spotify:track:(\w+)', selected_uri)
            if track_id_match:
                track_id = track_id_match.group(1)
                embed_url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator"
                # Use compact player height (80px)
                iframe_html = f'''<iframe style="border-radius:12px;" src="{embed_url}" width="100%" height="80" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>'''
                st.components.v1.html(iframe_html, height=80) # Adjust component height
                st.button("Clear Player ⏹️", key="clear_player", on_click=set_selected_track, args=(None,))
            else:
                if selected_uri is not None: st.warning(f"Invalid Track URI.")
                if st.session_state.selected_track_uri == selected_uri: st.session_state.selected_track_uri = None;
        else:
            st.caption("Click '▶️ Play' on a song row to load player.")
        st.markdown("---") # Visual separator

    st.header("⚙️ Filters & Options")
    # st.write("Apply filters to refine songs shown in 'Stage Explorer' and 'All Catalog'.") # Removed descriptive text
    # st.divider(); # Removed divider
    active_filters = {}
    # --- ADDED BACK General Filters ---
    st.subheader("General Filters")
    active_filters['song'] = st.text_input("Filter by Song Name (contains):")
    active_filters['artist'] = st.text_input("Filter by Artist Name (contains):")
    st.divider() # Add divider after general filters

    # --- Audio Feature Filters ---
    st.subheader("Audio Features") # Kept this subheader
    # Use FEATURES_FOR_UI for iteration, but filter out non-audio features
    audio_features_for_filtering = [f for f in FEATURES_FOR_UI if f not in ['stage_name', 'stage_number', 'stage_rank']]

    for feature in audio_features_for_filtering:
        feature_def = definitions.get(feature.lower(), f"Filter by {feature.replace('_',' ')}")

        if feature in CONTINUOUS_FEATURES: # Includes total_mismatch_score now
            min_val, max_val = feature_ranges.get(feature, (0.0, 1.0))
            # Use default range for score if needed
            if feature == 'total_mismatch_score':
                 min_val, max_val = feature_ranges.get(feature, (0.0, DEFAULT_MAX_SCORE))

            if min_val >= max_val:
                 if min_val == max_val: st.text(f"{feature.capitalize()}: {min_val}")
                 else: st.text(f"{feature.capitalize()}: Invalid range")
                 active_filters[feature] = None
            else:
                 f_min, f_max = float(min_val), float(max_val)
                 # Adjust step for score for better usability
                 step = 0.5 if feature == 'total_mismatch_score' else ((f_max - f_min) / 100 if (f_max - f_min) > 0 else 0.01)
                 # Set default value for score slider to max
                 default_value = f_max if feature == 'total_mismatch_score' else (f_min, f_max)
                 # Use single value slider for Max Score
                 if feature == 'total_mismatch_score':
                      active_filters[feature] = st.slider(f"Max Mismatch Score:", min_value=f_min, max_value=f_max, value=default_value, step=step, key=f"slider_{feature}", help=feature_def)
                 else:
                      active_filters[feature] = st.slider(f"{feature.replace('_',' ').capitalize()} Range:", min_value=f_min, max_value=f_max, value=default_value, step=step, key=f"slider_{feature}", help=feature_def)
        elif feature in DISCRETE_FEATURES: # Now only 'key'
            options = discrete_values.get(feature, []);
            if options: active_filters[feature] = st.multiselect(f"{feature.replace('_',' ').capitalize()} Select:", options=options, default=[], key=f"multi_{feature}", help=feature_def)
            else: active_filters[feature] = []

# --- Main Content Area (Controlled by Sidebar Radio) ---
# Display content based on the selected view in session state
current_view = st.session_state.current_view

# Ensure connection is available before proceeding
if not con:
     st.error("Database connection failed. Cannot display content.")
     st.stop()

# Check if data file path is set (either local or GCS for local testing)
if not DATA_PATH:
     st.error("Data path not defined in config.yaml. Stopping.")
     st.stop()


if current_view == "Stage Explorer":
    # st.header("🔍 Stage Explorer") # Removed header
    st.subheader("Explore Songs by Stage")
    stage_display_options = {num: f"{num} - {name}" for num, name in stage_mapping.items()}
    selected_stage_display = st.selectbox("Select a Stage:", options=list(stage_display_options.values()), index=0, key="explorer_stage_select")
    selected_stage_number = None
    for num, display_val in stage_display_options.items():
         if display_val == selected_stage_display: selected_stage_number = num; break
    if selected_stage_number is not None:
        # st.write(f"Displaying songs where **Stage {selected_stage_number} ({stage_mapping.get(selected_stage_number)})** is a top match, matching filters:") # Removed description
        total_rows = count_songs(con, selected_stage_number, tuple(sorted(active_filters.items())), feature_ranges)
        limit, offset = st.session_state.explorer_rows_per_page, (st.session_state.explorer_page - 1) * st.session_state.explorer_rows_per_page
        results_df = query_paged_songs(con, selected_stage_number, active_filters, limit, offset, feature_ranges)

        if not results_df.empty:
            # Add the action column
            results_df_display = results_df.copy()
            results_df_display.insert(0, "▶️ Play", False) # Add 'Play' column

            # Define columns for display
            display_cols_explorer = ["▶️ Play", 'stage_rank', 'song_name', 'artists', 'album', 'total_mismatch_score'] + ALL_AUDIO_FEATURES # Customize visible columns + Audio Features
            # Filter display_cols_explorer to only include columns present in the dataframe
            final_display_cols = [col for col in display_cols_explorer if col in results_df_display.columns]

            # Configure the data editor
            column_config = { "▶️ Play": st.column_config.CheckboxColumn("Play", default=False, width="small") }
            # Add format configs for audio features
            for feat in ALL_AUDIO_FEATURES:
                 if feat == 'loudness': column_config[feat] = st.column_config.NumberColumn(format="%.2f dB")
                 elif feat == 'tempo': column_config[feat] = st.column_config.NumberColumn(format="%.1f BPM")
                 elif feat == 'total_mismatch_score': column_config[feat] = st.column_config.NumberColumn(format="%.2f")
                 elif feat in CONTINUOUS_FEATURES: column_config[feat] = st.column_config.NumberColumn(format="%.3f")
                 # Hide columns not explicitly in display_cols_explorer but needed for selection
                 if feat not in final_display_cols: column_config[feat] = None

            column_config.update({"track_uri": None, "song_id": None, "stage_name": None, "stage_number": None})


            edited_df = st.data_editor(
                results_df_display[final_display_cols], # Display the df with the Play column
                use_container_width=True, hide_index=True,
                column_config=column_config, key="explorer_editor",
                disabled=list(results_df_display.columns.drop("▶️ Play")) # Make all columns except Play read-only
            )

            # Find which row was clicked (where '▶️ Play' changed to True)
            try:
                 # Compare edited_df with the version *before* adding the Play column
                 # Find rows where '▶️ Play' is True in edited_df
                 clicked_rows = edited_df[edited_df["▶️ Play"] == True]
                 if not clicked_rows.empty:
                      # Get the index from the *edited* dataframe
                      clicked_row_index = clicked_rows.index[0]
                      # Use this index to get the corresponding row from the *original* dataframe
                      selected_row_original = results_df.iloc[clicked_row_index]
                      selected_uri = selected_row_original.get('track_uri')
                      if selected_uri and st.session_state.selected_track_uri != selected_uri:
                           print(f"Explorer row selected via editor: {selected_uri}")
                           set_selected_track(selected_uri) # Use callback function
                           st.rerun() # Rerun to update player
            except IndexError:
                 print(f"IndexError during selection processing for explorer. Index might be invalid after rerun.")
            except Exception as e: st.error(f"Error processing table selection: {e}")

            # st.divider() # Removed divider
            display_pagination_controls(total_rows, key_prefix="explorer")
        else: st.info(f"No songs found matching the current filters for Stage {selected_stage_number}.")
    else: st.warning("Could not determine selected stage number.")

elif current_view == "All Catalog":
    # st.header("🎶 All Catalog") # Removed header
    st.subheader("Browse All Songs")
    # st.write("Showing all song rankings matching the selected filters.") # Removed description
    total_rows_catalog = count_songs(con, None, tuple(sorted(active_filters.items())), feature_ranges)
    limit_catalog, offset_catalog = st.session_state.catalog_rows_per_page, (st.session_state.catalog_page - 1) * st.session_state.catalog_rows_per_page
    results_df_catalog = query_paged_songs(con, None, active_filters, limit_catalog, offset_catalog, feature_ranges)

    if not results_df_catalog.empty:
        # Add the action column
        results_df_catalog_display = results_df_catalog.copy()
        results_df_catalog_display.insert(0, "▶️ Play", False)

        # Define columns to display in catalog view
        display_cols_catalog = ["▶️ Play", 'song_name', 'artists', 'album', 'stage_number', 'stage_name', 'stage_rank', 'total_mismatch_score'] + ALL_AUDIO_FEATURES # Customize + Audio Features
        # Filter display_cols_catalog to only include columns present in the dataframe
        final_display_cols_cat = [col for col in display_cols_catalog if col in results_df_catalog_display.columns]

        column_config_catalog = { "▶️ Play": st.column_config.CheckboxColumn("Play", default=False, width="small") }
        # Add format configs for audio features
        for feat in ALL_AUDIO_FEATURES:
             if feat == 'loudness': column_config_catalog[feat] = st.column_config.NumberColumn(format="%.2f dB")
             elif feat == 'tempo': column_config_catalog[feat] = st.column_config.NumberColumn(format="%.1f BPM")
             elif feat == 'total_mismatch_score': column_config_catalog[feat] = st.column_config.NumberColumn(format="%.2f")
             elif feat in CONTINUOUS_FEATURES: column_config_catalog[feat] = st.column_config.NumberColumn(format="%.3f")
             # Hide columns not explicitly in display_cols_catalog but needed for selection
             if feat not in final_display_cols_cat: column_config_catalog[feat] = None
        column_config_catalog.update({"track_uri": None, "song_id": None})


        edited_df_catalog = st.data_editor(
            results_df_catalog_display[final_display_cols_cat],
            use_container_width=True, hide_index=True,
            column_config=column_config_catalog, key="catalog_editor",
            disabled=list(results_df_catalog_display.columns.drop("▶️ Play"))
        )

        # Find which row was clicked
        try:
            diff_catalog = edited_df_catalog[edited_df_catalog["▶️ Play"] == True]
            if not diff_catalog.empty:
                clicked_row_index_cat = diff_catalog.index[0]
                selected_row_original_cat = results_df_catalog.iloc[clicked_row_index_cat]
                selected_uri_cat = selected_row_original_cat.get('track_uri')
                if selected_uri_cat and st.session_state.selected_track_uri != selected_uri_cat:
                    print(f"Catalog row selected via editor: {selected_uri_cat}")
                    set_selected_track(selected_uri_cat) # Use callback
                    st.rerun()
        except IndexError:
            print(f"IndexError during selection processing for catalog. Index: {clicked_row_index_cat}, DF length: {len(results_df_catalog)}")
        except Exception as e: st.error(f"Error processing table selection: {e}")

        # st.divider() # Removed divider
        display_pagination_controls(total_rows_catalog, key_prefix="catalog")
    else: st.info("No songs found matching the current filters.")

elif current_view == "Shuffle Journey":
    # st.header("🔀 Shuffle Journey") # Removed header
    st.subheader("Generate a Random Sound Journey")
    # st.write("Creates a 12-song sequence using the best available match for each stage.") # Removed description

    st.button(
        "🔀 Generate Shuffle Journey",
        key="generate_shuffle_button_pandas",
        on_click=run_shuffle_generation # Use callback
        )

    shuffle_results_area = st.container()
    with shuffle_results_area:
        if 'shuffle_df' in st.session_state and st.session_state.shuffle_df is not None:
            journey_df_to_display = st.session_state.shuffle_df
            fallback_used_to_display = st.session_state.shuffle_fallback
            print(f"[{time.time()}] Displaying shuffle data from session state. Fallback: {fallback_used_to_display}")

            if not journey_df_to_display.empty:
                if fallback_used_to_display: st.caption("ℹ️ _Note: For some stages, the top-ranked (Rank 1) song was not available, so the next best match was selected._")

                # Add the Play column for the editor
                journey_df_display_editor = journey_df_to_display.copy()
                journey_df_display_editor.insert(0, "▶️ Play", False)

                display_cols_shuffle = ["▶️ Play", 'stage', 'song_name', 'artists', 'album', 'stage_rank', 'total_mismatch_score'] # Customize visible
                actual_cols_shuffle = [col for col in display_cols_shuffle if col in journey_df_display_editor.columns]

                column_config_shuffle = {
                    "▶️ Play": st.column_config.CheckboxColumn("Play", default=False, width="small"),
                    "track_uri": None, "song_id": None, # Hide these
                    "total_mismatch_score": st.column_config.NumberColumn(format="%.2f"),
                }

                edited_df_shuffle = st.data_editor(
                    journey_df_display_editor[actual_cols_shuffle],
                    use_container_width=True, hide_index=True,
                    column_config=column_config_shuffle, key="shuffle_editor",
                    disabled=list(journey_df_display_editor.columns.drop("▶️ Play"))
                )

                # Find which row was clicked
                try:
                    diff_shuffle = edited_df_shuffle[edited_df_shuffle["▶️ Play"] == True]
                    if not diff_shuffle.empty:
                        clicked_row_index_shuf = diff_shuffle.index[0]
                        selected_row_original_shuf = journey_df_to_display.iloc[clicked_row_index_shuf]
                        selected_uri_shuf = selected_row_original_shuf.get('track_uri')
                        if selected_uri_shuf and st.session_state.selected_track_uri != selected_uri_shuf:
                            print(f"Shuffle row selected via editor: {selected_uri_shuf}")
                            set_selected_track(selected_uri_shuf) # Use callback
                            st.rerun()
                except IndexError:
                     print(f"IndexError during selection processing for shuffle. Index: {clicked_row_index_shuf}, DF length: {len(journey_df_to_display)}")
                except Exception as e: st.error(f"Error processing table selection: {e}")

            else: st.warning("Could not generate journey or no results found.")
        # else: st.info("Click the button above to generate a shuffle journey.")


elif current_view == "Spec Tester":
    # st.header("🧪 Spec Tester") # Removed header
    st.subheader("Test Song Classification with Current Specs")
    st.write(f"These calculations use the specifications currently loaded from `{CONFIG_FILE}`.")
    st.info(f"Edit the `config.yaml` file and **reload this page (F5 or Cmd+R)** in your browser to test different specs.")
    with st.expander("View Current Stage Specifications from config.yaml"): st.json(STAGE_SPECS)
    st.divider()
    st.session_state.test_song_id = st.text_input("Enter Song ID or partial name to test:", value=st.session_state.test_song_id, key="test_song_input", placeholder="e.g., 4iV5W9uYEdYUVa79Axb7Rh or Bohemian Rhapsody")
    if st.session_state.test_song_id:
        search_term = st.session_state.test_song_id.replace("'", "''")
        cols_to_select_test = ['id', 'name', 'artists'] + ALL_FEATURES_FROM_SPEC
        unique_cols_to_select_test = []; seen_test = set()
        for col in cols_to_select_test:
             if col.lower() not in seen_test:
                  try: con.execute(f"SELECT \"{col}\" FROM {read_from_source()} LIMIT 1;"); unique_cols_to_select_test.append(f"\"{col}\""); seen_test.add(col.lower())
                  except Exception: print(f"Warning: Column '{col}' not found in file for spec tester query."); pass
        if not unique_cols_to_select_test: st.error("Could not find required columns in the data file for testing."); st.stop()
        find_query = f"""SELECT {', '.join(unique_cols_to_select_test)} FROM {read_from_source()} WHERE id = '{search_term}' OR lower(name) LIKE lower('%{search_term}%') LIMIT 50;"""
        try:
            possible_songs_df = con.query(find_query).df()
            if not possible_songs_df.empty:
                if len(possible_songs_df) > 1:
                     possible_songs_df['display_option'] = possible_songs_df.apply(lambda row: f"{row['name']} by {row.get('artists','Unknown')} (ID: {row['id']})", axis=1)
                     selected_song_display = st.selectbox("Select the exact song:", options=possible_songs_df['display_option'].tolist(), index=0)
                     try: selected_song_id = selected_song_display.split("(ID: ")[1].replace(")", "")
                     except IndexError: st.error("Could not parse selected song ID."); st.stop()
                     song_features_series = possible_songs_df[possible_songs_df['id'] == selected_song_id].iloc[0]
                else:
                     song_features_series = possible_songs_df.iloc[0]; selected_song_id = song_features_series['id']
                     st.write(f"Found song: **{song_features_series['name']}** by {song_features_series.get('artists','Unknown')}")
                song_features_dict = {}
                for feature in ALL_FEATURES_FROM_SPEC:
                    actual_feature_col = next((col for col in song_features_series.index if col.lower() == feature.lower()), None)
                    if actual_feature_col: song_features_dict[feature] = song_features_series[actual_feature_col]
                    else: print(f"Warning: Feature '{feature}' from spec not found in song data for testing."); song_features_dict[feature] = None
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
        except Exception as e: st.error(f"Error finding or testing song: {e}"); st.error(f"Query attempted: {find_query}")

