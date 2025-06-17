import streamlit as st
import pandas as pd
import dask.dataframe as dd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Food Flow Impact Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #3b82f6, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3b82f6;
    }

    .impact-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    .stSelectbox > div > div {
        background-color: #f8fafc;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }

    /* Custom CSS for dark red delta text */
    [data-testid="metric-container"] [data-testid="metric-delta"] {
        color: #8B0000 !important;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """
    Load and prepare the food flow data with improved error handling and caching.
    """
    try:
        # Define file paths for deployment
        parquet_path = 'data/combined_commodity_df_allmods_pop.parquet'
        csv_path = 'data/demographics_economics_by_admin1region.csv'

        # Alternative paths to try if the main paths don't work
        alternative_paths = {
            'parquet': [
                './combined_commodity_df_allmods_pop.parquet',
                'combined_commodity_df_allmods_pop.parquet'
            ],
            'csv': [
                './demographics_economics_by_admin1region.csv',
                'demographics_economics_by_admin1region.csv'
            ]
        }

        # Try to load parquet file
        df = None
        for path in [parquet_path] + alternative_paths['parquet']:
            if os.path.exists(path):
                try:
                    # Use dask for large files, then compute to pandas
                    df_dask = dd.read_parquet(path)
                    df = df_dask.compute()
                    break
                except Exception as e:
                    try:
                        df = pd.read_parquet(path)
                        break
                    except Exception as e2:
                        continue

        if df is None:
            st.error("❌ Could not find or load the parquet file. Please check the file path.")
            return None, None, None

        # Try to load CSV file (full demographics data)
        df_demog = None
        for path in [csv_path] + alternative_paths['csv']:
            if os.path.exists(path):
                try:
                    df_demog = pd.read_csv(path)
                    break
                except Exception as e:
                    continue

        if df_demog is None:
            st.error("❌ Could not find or load the admin demographics CSV file.")
            return None, None, None

        # Extract admin names
        df_admin_name = df_demog[['ID', 'admin_name']].copy()

        # Data preprocessing and merging
        # Merge admin names with the main dataframe
        original_len = len(df)
        df = df.merge(
            df_admin_name,
            left_on='from_id',
            right_on='ID',
            how='left'
        ).rename(columns={'admin_name': 'from_id_admin_name'})

        # Drop the extra 'ID' column from the merge
        if 'ID' in df.columns:
            df.drop(columns=['ID'], inplace=True)

        # Display data summary
        st.success("✅ Data loaded successfully!")

        return df, df_admin_name, df_demog

    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None, None


def analyze_flow_impact(df, from_ids=None, to_id_pattern=None, remove_from_ids=None):
    """
    Analyze flows and estimate impact of removing specific from_id values.
    """
    # Convert single values to lists for consistent processing
    if from_ids is not None and not isinstance(from_ids, list):
        from_ids = [from_ids]
    if remove_from_ids is not None and not isinstance(remove_from_ids, list):
        remove_from_ids = [remove_from_ids]

    results = {}

    # 1. Filter data based on from_ids if specified
    if from_ids is not None:
        filtered_df = df[df['from_id'].isin(from_ids)].copy()
        results['analyzed_from_ids'] = from_ids
    else:
        filtered_df = df.copy()
        results['analyzed_from_ids'] = 'All'

    # 2. Filter data based on to_id pattern if specified
    if to_id_pattern is not None:
        filtered_df = filtered_df[filtered_df['to_id'].str.contains(to_id_pattern, na=False)]
        results['to_id_pattern'] = to_id_pattern
        results['matching_to_ids'] = sorted(filtered_df['to_id'].unique().tolist())
    else:
        results['to_id_pattern'] = 'All'
        results['matching_to_ids'] = 'All'

    # 3. Calculate current flows (baseline)
    baseline_totals = filtered_df.groupby('from_id').agg({
        'flow': 'sum',
        'Energy (kcal)': 'sum',
        'Protein (kg)': 'sum',
        'Fat (kg)': 'sum'
    }).reset_index()

    # Overall totals for the filtered data
    overall_baseline = {
        'total_flow': filtered_df['flow'].sum(),
        'total_energy': filtered_df['Energy (kcal)'].sum(),
        'total_protein': filtered_df['Protein (kg)'].sum(),
        'total_fat': filtered_df['Fat (kg)'].sum()
    }

    results['baseline_by_from_id'] = baseline_totals
    results['baseline_totals'] = overall_baseline

    # 4. Impact analysis if remove_from_ids is specified
    if remove_from_ids is not None:
        # Filter out the from_ids to be removed
        remaining_df = filtered_df[~filtered_df['from_id'].isin(remove_from_ids)].copy()

        # Calculate new totals after removal
        remaining_totals = {
            'total_flow': remaining_df['flow'].sum(),
            'total_energy': remaining_df['Energy (kcal)'].sum(),
            'total_protein': remaining_df['Protein (kg)'].sum(),
            'total_fat': remaining_df['Fat (kg)'].sum()
        }

        # Calculate removed amounts
        removed_amounts = {
            'removed_flow': overall_baseline['total_flow'] - remaining_totals['total_flow'],
            'removed_energy': overall_baseline['total_energy'] - remaining_totals['total_energy'],
            'removed_protein': overall_baseline['total_protein'] - remaining_totals['total_protein'],
            'removed_fat': overall_baseline['total_fat'] - remaining_totals['total_fat']
        }

        # Calculate percentage impact
        impact_percentages = {
            'flow_impact_pct': (removed_amounts['removed_flow'] / overall_baseline['total_flow'] * 100) if
            overall_baseline['total_flow'] > 0 else 0,
            'energy_impact_pct': (removed_amounts['removed_energy'] / overall_baseline['total_energy'] * 100) if
            overall_baseline['total_energy'] > 0 else 0,
            'protein_impact_pct': (removed_amounts['removed_protein'] / overall_baseline['total_protein'] * 100) if
            overall_baseline['total_protein'] > 0 else 0,
            'fat_impact_pct': (removed_amounts['removed_fat'] / overall_baseline['total_fat'] * 100) if
            overall_baseline['total_fat'] > 0 else 0
        }

        # Detailed breakdown of what was removed
        removed_details = filtered_df[filtered_df['from_id'].isin(remove_from_ids)].groupby('from_id').agg({
            'flow': 'sum',
            'Energy (kcal)': 'sum',
            'Protein (kg)': 'sum',
            'Fat (kg)': 'sum'
        }).reset_index()

        results['removal_analysis'] = {
            'removed_from_ids': remove_from_ids,
            'remaining_totals': remaining_totals,
            'removed_amounts': removed_amounts,
            'impact_percentages': impact_percentages,
            'removed_details_by_from_id': removed_details
        }

    # 5. Summary statistics
    results['summary'] = {
        'total_unique_from_ids': len(filtered_df['from_id'].unique()),
        'total_unique_to_ids': len(filtered_df['to_id'].unique()),
        'total_records': len(filtered_df)
    }

    return results


def analyze_subnational_impact(df, admin_label, remove_admin_ids, target_admin_patterns, flow_types=None):
    """
    Analyze impact of removing specific subnational admin regions.
    """
    # Filter by flow types if specified
    if flow_types is not None:
        df_filtered = df[df['flow_type'].isin(flow_types)].copy()
    else:
        df_filtered = df.copy()

    # Get admin names for removed regions
    remove_admin_ids_label = admin_label[admin_label['ID'].isin(remove_admin_ids)]['admin_name'].tolist()

    # 1. GLOBAL IMPACT
    global_results = analyze_flow_impact(df_filtered, remove_from_ids=remove_admin_ids)

    # 2. IMPACT ON TARGET REGIONS (using your correct approach)
    target_results = []

    for pattern in target_admin_patterns:
        # Find all admin IDs matching the pattern
        matching_to_ids = df_filtered[df_filtered['to_id'].str.contains(pattern, na=False)]['to_id'].unique()

        if len(matching_to_ids) > 0:
            # This is the key: filter to target pattern FIRST, then analyze impact
            target_impact = analyze_flow_impact(df_filtered, to_id_pattern=pattern, remove_from_ids=remove_admin_ids)

            if 'removal_analysis' in target_impact and target_impact['baseline_totals']['total_energy'] > 0:
                removed = target_impact['removal_analysis']['removed_amounts']
                impact_pct = target_impact['removal_analysis']['impact_percentages']
                baseline = target_impact['baseline_totals']

                target_results.append({
                    'Target_Pattern': pattern,
                    'Matching_Admin_IDs': len(matching_to_ids),
                    'Current_Energy_kcal': baseline['total_energy'],
                    'Lost_Energy_kcal': removed['removed_energy'],
                    'Energy_Loss_Pct': impact_pct['energy_impact_pct'],
                    'Current_Protein_kg': baseline['total_protein'],
                    'Lost_Protein_kg': removed['removed_protein'],
                    'Protein_Loss_Pct': impact_pct['protein_impact_pct'],
                    'Current_Fat_kg': baseline['total_fat'],
                    'Lost_Fat_kg': removed['removed_fat'],
                    'Fat_Loss_Pct': impact_pct['fat_impact_pct']
                })
            else:
                target_results.append({
                    'Target_Pattern': pattern,
                    'Matching_Admin_IDs': len(matching_to_ids),
                    'Current_Energy_kcal': 0,
                    'Lost_Energy_kcal': 0,
                    'Energy_Loss_Pct': 0,
                    'Current_Protein_kg': 0,
                    'Lost_Protein_kg': 0,
                    'Protein_Loss_Pct': 0,
                    'Current_Fat_kg': 0,
                    'Lost_Fat_kg': 0,
                    'Fat_Loss_Pct': 0
                })
        else:
            target_results.append({
                'Target_Pattern': pattern,
                'Matching_Admin_IDs': 0,
                'Current_Energy_kcal': 0,
                'Lost_Energy_kcal': 0,
                'Energy_Loss_Pct': 0,
                'Current_Protein_kg': 0,
                'Lost_Protein_kg': 0,
                'Protein_Loss_Pct': 0,
                'Current_Fat_kg': 0,
                'Lost_Fat_kg': 0,
                'Fat_Loss_Pct': 0
            })

    return {
        'removed_regions': remove_admin_ids_label,
        'global_results': global_results,
        'target_results': target_results,
        'flow_types': flow_types
    }


def format_large_number(number):
    """
    Format large numbers with appropriate suffixes (K, M, B, T)
    """
    if pd.isna(number) or number == 0:
        return "0"

    abs_number = abs(number)

    if abs_number >= 1e12:
        return f"{number / 1e12:.1f}T"
    elif abs_number >= 1e9:
        return f"{number / 1e9:.1f}B"
    elif abs_number >= 1e6:
        return f"{number / 1e6:.1f}M"
    elif abs_number >= 1e3:
        return f"{number / 1e3:.1f}K"
    else:
        return f"{number:.0f}"


def format_percentage(number):
    """Format percentage with appropriate decimal places"""
    if pd.isna(number):
        return "0.00%"
    return f"{number:.2f}%"


def calculate_people_impact(energy_lost, protein_lost, fat_lost, daily_energy_req=2000, daily_protein_req_kg=0.05,
                            daily_fat_req_kg=0.065, time_period_days=365):
    """
    Calculate number of people impacted based on daily nutritional requirements.

    Parameters:
    - energy_lost: kcal lost (total amount in dataset time period)
    - protein_lost: kg lost (total amount in dataset time period)
    - fat_lost: kg lost (total amount in dataset time period)
    - daily_energy_req: kcal per person per day (default: 2000)
    - daily_protein_req_kg: kg per person per day (default: 0.05 = 50g)
    - daily_fat_req_kg: kg per person per day (default: 0.065 = 65g)
    - time_period_days: number of days the lost amounts represent (default: 365 for annual)

    Returns:
    - Dictionary with people impact calculations
    """

    # Calculate total nutritional needs per person for the time period
    total_energy_per_person = daily_energy_req * time_period_days
    total_protein_per_person = daily_protein_req_kg * time_period_days
    total_fat_per_person = daily_fat_req_kg * time_period_days

    # Calculate number of people affected over the time period
    people_energy = energy_lost / total_energy_per_person if energy_lost > 0 else 0
    people_protein = protein_lost / total_protein_per_person if protein_lost > 0 else 0
    people_fat = fat_lost / total_fat_per_person if fat_lost > 0 else 0

    return {
        'people_energy': people_energy,
        'people_protein': people_protein,
        'people_fat': people_fat,
        'time_period_days': time_period_days,
        'daily_energy_req': daily_energy_req,
        'daily_protein_req_kg': daily_protein_req_kg,
        'daily_fat_req_kg': daily_fat_req_kg
    }


def create_impact_chart(data, title, metric_type):
    """Create impact visualization chart with smart number formatting"""
    fig = go.Figure()

    # Add bars for current vs lost amounts
    fig.add_trace(go.Bar(
        name=f'Current {metric_type}',
        x=data['Target_Pattern'],
        y=data[f'Current_{metric_type}'],
        marker_color='#3b82f6',
        text=[format_large_number(val) for val in data[f'Current_{metric_type}']],
        textposition='auto',
    ))

    fig.add_trace(go.Bar(
        name=f'Lost {metric_type}',
        x=data['Target_Pattern'],
        y=data[f'Lost_{metric_type}'],
        marker_color='#ef4444',
        text=[format_large_number(val) for val in data[f'Lost_{metric_type}']],
        textposition='auto',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Target Regions",
        yaxis_title=metric_type,
        barmode='group',
        height=400,
        template='plotly_white',
        showlegend=True
    )

    return fig


def create_people_impact_chart(data, title):
    """Create people impact visualization chart with smart number formatting"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='People Energy Impact',
        x=data['Target_Pattern'],
        y=data['People_Energy'],
        marker_color='#f59e0b',
        text=[format_large_number(val) for val in data['People_Energy']],
        textposition='auto',
    ))

    fig.add_trace(go.Bar(
        name='People Protein Impact',
        x=data['Target_Pattern'],
        y=data['People_Protein'],
        marker_color='#10b981',
        text=[format_large_number(val) for val in data['People_Protein']],
        textposition='auto',
    ))

    fig.add_trace(go.Bar(
        name='People Fat Impact',
        x=data['Target_Pattern'],
        y=data['People_Fat'],
        marker_color='#8b5cf6',
        text=[format_large_number(val) for val in data['People_Fat']],
        textposition='auto',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Target Regions",
        yaxis_title="Number of People Affected",
        barmode='group',
        height=400,
        template='plotly_white',
        showlegend=True
    )

    return fig


def create_percentage_chart(data, title):
    """Create percentage impact chart with energy, protein, and fat"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Energy Loss %',
        x=data['Target_Pattern'],
        y=data['Energy_Loss_Pct'],
        marker_color='#f59e0b',
        text=[f"{val:.2f}%" for val in data['Energy_Loss_Pct']],
        textposition='auto',
    ))

    fig.add_trace(go.Bar(
        name='Protein Loss %',
        x=data['Target_Pattern'],
        y=data['Protein_Loss_Pct'],
        marker_color='#10b981',
        text=[f"{val:.2f}%" for val in data['Protein_Loss_Pct']],
        textposition='auto',
    ))

    fig.add_trace(go.Bar(
        name='Fat Loss %',
        x=data['Target_Pattern'],
        y=data['Fat_Loss_Pct'],
        marker_color='#8b5cf6',
        text=[f"{val:.2f}%" for val in data['Fat_Loss_Pct']],
        textposition='auto',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Target Regions",
        yaxis_title="Percentage Loss (%)",
        barmode='group',
        height=400,
        template='plotly_white',
        showlegend=True
    )

    return fig


# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 Food Flow Impact Tool</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar for inputs
    st.sidebar.header("🔧 Configuration")

    # Load the real data
    with st.spinner("Loading data files..."):
        df, df_admin_name, df_demog = load_data()
        if df is None or df_admin_name is None or df_demog is None:
            st.error("Failed to load data files. Please check file paths and try again.")
            st.stop()

        st.session_state.df = df
        st.session_state.df_admin_name = df_admin_name
        st.session_state.df_demog = df_demog

    df = st.session_state.df
    df_admin_name = st.session_state.df_admin_name
    df_demog = st.session_state.df_demog

    # Region selection
    st.sidebar.subheader("🗺️ Region Selection")

    # Create unique admin_name, country combinations
    admin_country_df = df_admin_name.merge(
        df_demog[['ID', 'country']],
        on='ID',
        how='left'
    ).dropna()

    # Create display names: "Admin Name, Country"
    admin_country_df['display_name'] = admin_country_df['admin_name'] + ', ' + admin_country_df['country']
    admin_country_df = admin_country_df.drop_duplicates(subset=['display_name'])

    # Get all unique display names
    all_display_names = sorted(admin_country_df['display_name'].tolist())

    # Create display name to ID mapping
    region_mapping = dict(zip(admin_country_df['display_name'], admin_country_df['ID']))

    # Default selections
    default_regions = ['Paraná, Brazil', 'Uttar Pradesh, India', 'Western Australia, Australia']
    # Filter defaults to only include regions that actually exist in the data
    available_defaults = [region for region in default_regions if region in all_display_names]
    if not available_defaults:
        available_defaults = all_display_names[:3] if len(all_display_names) >= 3 else all_display_names

    # Multiselect with admin name, country combinations
    selected_region_names = st.sidebar.multiselect(
        "Select regions to remove",
        options=all_display_names,
        default=available_defaults,
        help="Choose the admin regions to remove for impact analysis. Format: Admin Name, Country. You can search by typing part of the region or country name."
    )

    # Convert selected display names to IDs
    remove_regions = [region_mapping[name] for name in selected_region_names if name in region_mapping]

    # Target region selection using countries
    st.sidebar.subheader("🎯 Target Regions")

    # Get unique countries from demographics data
    if 'country' in df_demog.columns and 'iso3' in df_demog.columns:
        country_df = df_demog[['country', 'iso3']].drop_duplicates().dropna()
        country_mapping = dict(zip(country_df['country'], country_df['iso3']))
        available_countries = sorted(country_df['country'].unique().tolist())

        # Default country selections
        default_countries = ['United States', 'China', 'Mexico']
        available_default_countries = [country for country in default_countries if country in available_countries]
        if not available_default_countries:
            available_default_countries = available_countries[:3] if len(
                available_countries) >= 3 else available_countries

        selected_countries = st.sidebar.multiselect(
            "Select target countries",
            options=available_countries,
            default=available_default_countries,
            help="Choose countries to analyze the impact on. Multiple countries can be selected."
        )

        # Convert country names to ISO3 codes for pattern matching
        target_patterns = [country_mapping[country] for country in selected_countries if country in country_mapping]
    else:
        # Fallback to text input if country data not available
        st.sidebar.warning("Country data not found in demographics file. Using manual input.")
        target_input = st.sidebar.text_input(
            "Target region patterns (comma-separated)",
            value="USA,CHN,MEX",
            help="Enter country codes or patterns to match target regions"
        )
        target_patterns = [p.strip() for p in target_input.split(',') if p.strip()]

    # Add a separator
    st.sidebar.markdown("---")

    # Custom nutritional requirements
    st.sidebar.subheader("🍽️ Daily Nutritional Requirements")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        daily_energy_req = st.number_input(
            "Energy (kcal/person/day)",
            min_value=500,
            max_value=5000,
            value=2000,
            step=100,
            help="Daily energy requirement per person"
        )

        daily_protein_req = st.number_input(
            "Protein (g/person/day)",
            min_value=10,
            max_value=200,
            value=50,
            step=5,
            help="Daily protein requirement per person"
        )

    with col2:
        daily_fat_req = st.number_input(
            "Fat (g/person/day)",
            min_value=10,
            max_value=200,
            value=65,
            step=5,
            help="Daily fat requirement per person"
        )

        # Convert to kg for calculations
        daily_protein_req_kg = daily_protein_req / 1000
        daily_fat_req_kg = daily_fat_req / 1000

    st.sidebar.info(
        f"Current settings: {daily_energy_req} kcal, {daily_protein_req}g protein, {daily_fat_req}g fat per person per day")

    # Add a separator
    st.sidebar.markdown("---")

    # Flow type selection
    if 'flow_type' in df.columns:
        st.sidebar.subheader("⚡ Flow Types")

        # Map original flow types to readable names
        flow_type_mapping = {
            'sea_re': 'Maritime International Re-exports',
            'sea_dom': 'Maritime International Flows',
            'land_re': 'Land International Re-exports',
            'land_dom': 'Land International Flows',
            'within': 'Within Country'
        }

        available_flow_types = df['flow_type'].unique().tolist()

        # Create readable options
        readable_options = []
        for flow_type in available_flow_types:
            if flow_type in flow_type_mapping:
                readable_options.append(flow_type_mapping[flow_type])
            else:
                readable_options.append(flow_type)

        # Create reverse mapping for converting back
        reverse_mapping = {v: k for k, v in flow_type_mapping.items()}

        selected_flow_types_readable = st.sidebar.multiselect(
            "Select flow types (optional)",
            options=readable_options,
            default=[],  # No default selection
            help="Choose specific flow types to analyze. If none selected, all flow types will be used."
        )

        # Convert back to original flow type codes
        if selected_flow_types_readable:
            flow_types = []
            for readable_type in selected_flow_types_readable:
                if readable_type in reverse_mapping:
                    flow_types.append(reverse_mapping[readable_type])
                else:
                    # If it's not in our mapping, it might be an original type
                    flow_types.append(readable_type)
        else:
            flow_types = None  # Use all flow types

        # Display selection info
        if flow_types:
            st.sidebar.write("**Selected flow types:**")
            for flow_type in flow_types:
                readable_name = flow_type_mapping.get(flow_type, flow_type)
                st.sidebar.write(f"• {readable_name}")
        else:
            st.sidebar.info("All flow types will be used")
    else:
        flow_types = None

    # Analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        if not remove_regions:
            st.error("Please select at least one region to remove.")
            return

        if not target_patterns:
            st.error("Please specify at least one target pattern.")
            return

        # Run analysis
        with st.spinner("Running impact analysis..."):
            try:
                results = analyze_subnational_impact(
                    df, df_admin_name, remove_regions, target_patterns, flow_types
                )
                st.session_state.results = results
                st.session_state.daily_reqs = {
                    'energy': daily_energy_req,
                    'protein_kg': daily_protein_req_kg,
                    'fat_kg': daily_fat_req_kg,
                    'time_period_days': 365  # Fixed to 365 days (annual)
                }
            except Exception as e:
                st.error(f"Error running analysis: {str(e)}")
                return

    # Display results
    if 'results' in st.session_state:
        results = st.session_state.results
        daily_reqs = st.session_state.get('daily_reqs', {
            'energy': 2000,
            'protein_kg': 0.05,
            'fat_kg': 0.065,
            'time_period_days': 365  # Fixed to 365 days
        })

        # Global Impact
        st.header("🌍 Global Impact")
        global_results = results['global_results']

        if 'removal_analysis' in global_results:
            removed = global_results['removal_analysis']['removed_amounts']
            impact_pct = global_results['removal_analysis']['impact_percentages']
            baseline = global_results['baseline_totals']

            # Calculate global people impact
            global_people_impact = calculate_people_impact(
                removed['removed_energy'],
                removed['removed_protein'],
                removed['removed_fat'],
                daily_reqs['energy'],
                daily_reqs['protein_kg'],
                daily_reqs['fat_kg'],
                daily_reqs['time_period_days']
            )

            # Global impact metrics - first row
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Flow Impact",
                    format_percentage(impact_pct['flow_impact_pct']),
                    f"{format_large_number(removed['removed_flow'])} units lost",
                    delta_color="off"
                )

            with col2:
                st.metric(
                    "Energy Impact",
                    format_percentage(impact_pct['energy_impact_pct']),
                    f"{format_large_number(removed['removed_energy'])} kcal lost",
                    delta_color="off"
                )

            with col3:
                st.metric(
                    "Protein Impact",
                    format_percentage(impact_pct['protein_impact_pct']),
                    f"{format_large_number(removed['removed_protein'])} kg lost",
                    delta_color="off"
                )

            with col4:
                st.metric(
                    "Fat Impact",
                    format_percentage(impact_pct['fat_impact_pct']),
                    f"{format_large_number(removed['removed_fat'])} kg lost",
                    delta_color="off"
                )

            # Global people impact metrics - second row
            st.subheader("👥 Global People Impact")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "People Energy Impact",
                    format_large_number(global_people_impact['people_energy']),
                    f"People affected (annual)",
                    delta_color="off"
                )

            with col2:
                st.metric(
                    "People Protein Impact",
                    format_large_number(global_people_impact['people_protein']),
                    f"People affected (annual)",
                    delta_color="off"
                )

            with col3:
                st.metric(
                    "People Fat Impact",
                    format_large_number(global_people_impact['people_fat']),
                    f"People affected (annual)",
                    delta_color="off"
                )

        # Impact on Target Countries
        if results['target_results']:
            st.header("🎯 Impact on Target Countries")

            target_df = pd.DataFrame(results['target_results'])

            # Calculate people impact for each target region using custom requirements
            people_impact_data = []
            for _, row in target_df.iterrows():
                people_impact = calculate_people_impact(
                    row['Lost_Energy_kcal'],
                    row['Lost_Protein_kg'],
                    row['Lost_Fat_kg'] if 'Lost_Fat_kg' in row else 0,
                    daily_reqs['energy'],
                    daily_reqs['protein_kg'],
                    daily_reqs['fat_kg'],
                    daily_reqs['time_period_days']
                )
                people_impact_data.append({
                    'Target_Pattern': row['Target_Pattern'],
                    'People_Energy': people_impact['people_energy'],
                    'People_Protein': people_impact['people_protein'],
                    'People_Fat': people_impact['people_fat']
                })

            people_df = pd.DataFrame(people_impact_data)

            # REVERSED ORDER: Percentage chart FIRST, then People analysis
            # Percentage impact chart (now includes fat)
            percentage_chart = create_percentage_chart(target_df, "Percentage Impact by Region")
            st.plotly_chart(percentage_chart, use_container_width=True)

            # People Impact Visualization (MOVED TO BOTTOM)
            st.subheader("👥 People Impact Analysis")
            st.write(
                f"*Based on annual nutritional requirements: {daily_reqs['energy']} kcal energy, {daily_reqs['protein_kg'] * 1000:.0f}g protein, {daily_reqs['fat_kg'] * 1000:.0f}g fat per person per day*")

            # Add interpretation note
            st.info(f"""
            **📊 Interpretation:** These numbers show how many people's **annual nutritional needs** 
            would be affected by the food flow disruption.
            """)

            people_chart = create_people_impact_chart(people_df, f"People Affected by Annual Nutritional Requirements")
            st.plotly_chart(people_chart, use_container_width=True)

            # Key insights
            st.subheader("🔍 Key Insights")

            # Find most impacted region by people
            max_people_energy_idx = people_df['People_Energy'].idxmax()
            max_people_protein_idx = people_df['People_Protein'].idxmax()

            col1, col2 = st.columns(2)

            with col1:
                st.success(f"""
                **Most Energy-Impacted Region (People):**
                - **{people_df.loc[max_people_energy_idx, 'Target_Pattern']}** affects {format_large_number(people_df.loc[max_people_energy_idx, 'People_Energy'])} people's daily energy needs
                - **{format_percentage(target_df.loc[max_people_energy_idx, 'Energy_Loss_Pct'])}** of energy supply lost
                """)

            with col2:
                st.warning(f"""
                **Most Protein-Impacted Region (People):**
                - **{people_df.loc[max_people_protein_idx, 'Target_Pattern']}** affects {format_large_number(people_df.loc[max_people_protein_idx, 'People_Protein'])} people's daily protein needs
                - **{format_percentage(target_df.loc[max_people_protein_idx, 'Protein_Loss_Pct'])}** of protein supply lost
                """)

        else:
            st.warning("No target region impacts found. Please check your target patterns.")

    else:
        # Welcome message
        st.info("""
        👋 **Welcome to the Food Flow Impact Analysis Tool!**

        This application helps you analyze the impact of removing specific subnational administrative regions 
        on global and target food flows. 

        **To get started:**
        1. Configure your analysis parameters in the sidebar
        2. Select regions to remove and target countries to analyze
        3. Click "Run Analysis" to see the results

        **Your real data will be loaded automatically.**
        """)


if __name__ == "__main__":
    main()