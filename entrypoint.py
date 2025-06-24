#entrypoint.py
#BuiltWithLove by @papitx0
import time
import os
from user import render_user_login_page
import datetime
import plotly.express as px
import streamlit as st
import pandas as pd
import re
import numpy as np
from difflib import SequenceMatcher
import threading
from web import ANPRParkingIntegration




st.set_page_config(
    page_title="BASE-SmartPark",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from datetime import datetime
from web import (
    ParkingDatabase,
    render_enhanced_reservation_page,
    render_anpr_dashboard,
    get_anpr_integration
)
from admin import (
    render_analytics_page,
    render_system_settings_page,
    render_admin_spot_map,
    render_user_admin_panel,
    render_user_passwords_view
)


# Cache database
@st.cache_resource
def get_db():
    return ParkingDatabase()


# Cache ANPR integration
@st.cache_resource
def get_anpr():
    return get_anpr_integration()


# Init session state
def init_session():
    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False
    if 'admin_username' not in st.session_state:
        st.session_state.admin_username = ""
    if 'page_refresh' not in st.session_state:
        st.session_state.page_refresh = 0
    if 'user_plate' not in st.session_state:
        st.session_state.user_plate = ""



def render_dashboard_page(db):
    st.header("🏠 SmartPark Dashboard")

    # Get current data with error handling
    try:
        spots_df = db.get_parking_spots()
        reservations_df = db.get_reservations_history()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    # Check if data exists
    if spots_df.empty:
        st.warning("No parking spot data available.")
        return

    # Summary statistics at the top
    st.subheader("📊 Parking Summary")

    # Calculate counts with better handling
    status_counts = spots_df['status'].value_counts()
    available_count = status_counts.get('available', 0)
    occupied_count = status_counts.get('occupied', 0)
    reserved_count = status_counts.get('reserved', 0)
    total_spots = len(spots_df)

    # Display enhanced summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🟢 Available",
            available_count,
            delta=f"{(available_count / total_spots * 100):.1f}%" if total_spots > 0 else "0%"
        )

    with col2:
        st.metric(
            "🔴 Occupied",
            occupied_count,
            delta=f"{(occupied_count / total_spots * 100):.1f}%" if total_spots > 0 else "0%"
        )

    with col3:
        st.metric(
            "🟡 Reserved",
            reserved_count,
            delta=f"{(reserved_count / total_spots * 100):.1f}%" if total_spots > 0 else "0%"
        )

    with col4:
        st.metric("🏢 Total Spots", total_spots)

    # Add occupancy rate visualization
    if total_spots > 0:
        occupancy_rate = (occupied_count + reserved_count) / total_spots
        st.progress(occupancy_rate, text=f"Occupancy Rate: {occupancy_rate:.1%}")

    # Filter options
    st.subheader("🔍 Spot Details")

    # Status filter
    status_options = ['All'] + list(spots_df['status'].unique())
    selected_status = st.selectbox("Filter by Status:", status_options, index=1 if 'available' in status_options else 0)

    # Apply filter
    if selected_status == 'All':
        filtered_spots = spots_df
        table_title = "All Parking Spots"
    else:
        filtered_spots = spots_df[spots_df['status'] == selected_status]
        table_title = f"{selected_status.title()} Parking Spots"

    st.subheader(f"🅿️ {table_title}")

    if not filtered_spots.empty:
        # Enhanced dataframe with better formatting
        display_df = filtered_spots.copy()

        # Format the dataframe for better display
        if any('time' in col.lower() or 'date' in col.lower() for col in display_df.columns):
            time_cols = [col for col in display_df.columns if 'time' in col.lower() or 'date' in col.lower()]
            for col in time_cols:
                display_df[col] = pd.to_datetime(display_df[col], errors='coerce')

        # Create column config based on available columns
        column_config = {}

        for col in display_df.columns:
            if col == "spot_id":
                column_config[col] = st.column_config.TextColumn(
                    "Spot ID",
                    help="Unique identifier for the parking spot"
                )
            elif col == "status":
                column_config[col] = st.column_config.TextColumn(
                    "Status",
                    help="Current status of the parking spot"
                )
            elif col == "current_user" or col == "user_id":
                column_config[col] = st.column_config.TextColumn(
                    "Current User",
                    help="User currently occupying or reserving the spot"
                )
            elif 'time' in col.lower() or 'date' in col.lower():
                column_config[col] = st.column_config.DatetimeColumn(
                    col.replace('_', ' ').title(),
                    format="YYYY-MM-DD HH:mm:ss",
                    help=f"Timestamp for {col.replace('_', ' ')}"
                )

        st.dataframe(
            display_df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True
        )

        # Add quick stats for filtered view
        if selected_status != 'All':
            st.info(f"Showing {len(filtered_spots)} {selected_status} spots out of {total_spots} total spots")
    else:
        st.info(f"No {selected_status.lower()} spots found.")

    # Recent activity section





# admin login function
def render_admin_login_page():
    st.header("🔐 Admin Login")

    if st.session_state.admin_logged_in:
        st.success(f"✅ Logged in as: {st.session_state.admin_username}")
        if st.button("🚪 Logout"):
            st.session_state.admin_logged_in = False
            st.session_state.admin_username = ""
            st.rerun()
        return

    with st.form("admin_login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.form_submit_button("🔑 Login"):
            # Simple admin check (you can enhance this with your database)
            if username == "admin" and password == "admin123":
                st.session_state.admin_logged_in = True
                st.session_state.admin_username = username
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")


# Reservation status tracker (User)
def _activate_reservation_from_anpr(db, reservation):
    """Activate a reservation when ANPR detects the vehicle."""
    try:
        # Get the current reservations data
        df = db.get_reservations_history()

        # Find the specific reservation to activate
        mask = (df['plate_number'].str.upper() == reservation['plate_number'].upper()) & \
               (df['created_at'] == reservation['created_at'])

        # Update reservation status and timing (fixed 15 minutes)
        now = datetime.now()
        end_time = now + pd.Timedelta(minutes=15)

        df.loc[mask, 'status'] = 'active'
        df.loc[mask, 'start_time'] = now.strftime('%Y-%m-%d %H:%M:%S')
        df.loc[mask, 'end_time'] = end_time.strftime('%Y-%m-%d %H:%M:%S')

        # Update spot status to occupied
        db.update_spot_status(reservation['spot_id'], 'occupied')

        # Save the updated reservations
        df.to_csv(db.reservations_file, index=False)

        return True

    except Exception as e:
        st.error(f"Error activating reservation: {e}")
        return False
def render_reservation_status_page(db):
    """Render the reservation status tracking page with improved UX and error handling."""
    st.header("📟 Reservation Status Tracker")

    # Input section with validation
    plate = st.text_input(
        "🔍 Enter your license plate to track reservation",
        placeholder="e.g., ABC-123",
        help="Enter your complete license plate number"
    )

    if plate and plate.strip():
        normalized_plate = plate.strip().upper()
        st.session_state.user_plate = normalized_plate
    elif plate == "":
        # Clear session state when input is cleared
        st.session_state.user_plate = None

    # Only proceed if we have a valid plate
    if not st.session_state.get('user_plate'):
        st.info("👆 Please enter your license plate above to track your reservation")
        return

    try:
        # Fetch and filter reservations
        df = db.get_reservations_history()
        if df.empty:
            st.warning("No reservations found in the system.")
            return

        user_reservations = df[df['plate_number'].str.upper() == st.session_state.user_plate]

        if user_reservations.empty:
            st.info(f"🔍 No reservations found for plate **{st.session_state.user_plate}**")
            st.caption("Double-check your license plate number and try again.")
            return

        # Get the most recent reservation
        latest_reservation = user_reservations.sort_values("created_at", ascending=False).iloc[0]

        # Display reservation details
        _display_reservation_header(latest_reservation)

        # Handle different status cases
        status = latest_reservation['status'].lower()

        if status == "waiting_detection":
            _handle_waiting_status(db, latest_reservation)
        elif status == "active":
            _handle_active_status(latest_reservation)
        elif status == "cancelled":
            _handle_cancelled_status(latest_reservation)
        elif status == "expired":
            _handle_expired_status(latest_reservation)
        elif status == "completed":
            _handle_completed_status(latest_reservation)
        else:
            st.info(f"ℹ️ Reservation status: **{status.title()}**")

        # Show reservation history if multiple exist
        if len(user_reservations) > 1:
            _display_reservation_history(user_reservations)

    except Exception as e:
        st.error("❌ Error retrieving reservation data. Please try again later.")
        st.exception(e) if st.session_state.get('debug_mode') else None


def _display_reservation_header(reservation):
    """Display the main reservation information in a clean format."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🅿️ Parking Spot", reservation['spot_id'])

    with col2:
        st.metric("👤 Customer", reservation['customer_name'])

    with col3:
        st.metric("⏱️ Duration", f"{reservation['duration_minutes']} min")

    # Status badge
    status = reservation['status'].upper()
    status_colors = {
        'WAITING_DETECTION': '🟡',
        'ACTIVE': '🟢',
        'CANCELLED': '🔴',
        'EXPIRED': '🟠',
        'COMPLETED': '🔵'
    }

    status_icon = status_colors.get(status, '⚪')
    st.markdown(f"### {status_icon} Status: **{status.replace('_', ' ').title()}**")


def _handle_waiting_status(db, reservation):
    """Handle reservations waiting for ANPR detection."""
    created_at = pd.to_datetime(reservation['created_at'], errors='coerce')
    if pd.isna(created_at):
        st.error("Invalid reservation timestamp")
        return

    total_waiting_seconds = (datetime.now() - created_at).total_seconds()
    minutes_waiting = int(total_waiting_seconds // 60)
    seconds_waiting = int(total_waiting_seconds % 60)

    timeout_minutes = 30
    remaining_seconds = max(0, (timeout_minutes * 60) - total_waiting_seconds)
    remaining_minutes = int(remaining_seconds // 60)
    remaining_sec_part = int(remaining_seconds % 60)

    # Check if ANPR has detected this vehicle
    anpr_detected = _check_anpr_detection(db, reservation['plate_number'])
    if anpr_detected:
        # ANPR detected the vehicle - activate the reservation
        _activate_reservation_from_anpr(db, reservation)
        st.success("✅ **Vehicle Detected!** Your reservation is now active.")
        st.rerun()
        return

    if remaining_seconds > 0:
        st.warning(f"⏳ **Waiting for vehicle detection...**")
        st.progress((minutes_waiting * 60 + seconds_waiting) / (timeout_minutes * 60))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("⏰ Time Waiting", f"{minutes_waiting}m {seconds_waiting}s")
        with col2:
            st.metric("⏱️ Time Remaining", f"{remaining_minutes}m {remaining_sec_part}s")

        st.info(
            "💡 **Important:** Drive to your assigned spot within 30 minutes or your reservation will be automatically cancelled."
        )
        st.caption("🎥 Our ANPR system will automatically detect your vehicle when you arrive.")

        # Auto-refresh every 10 seconds
        time.sleep(1)
        st.rerun()
    else:
        # Cancel expired reservation
        _cancel_expired_reservation(db, reservation)


def _cancel_expired_reservation(db, reservation):
    """Cancel a reservation that exceeded the waiting timeout."""
    try:
        df = db.get_reservations_history()
        mask = (df['plate_number'].str.upper() == reservation['plate_number'].upper()) & \
               (df['created_at'] == reservation['created_at'])

        df.loc[mask, 'status'] = 'cancelled'
        db.update_spot_status(reservation['spot_id'], 'available')
        df.to_csv(db.reservations_file, index=False)

        st.error("❌ **Reservation Cancelled**")
        st.markdown("Your reservation was cancelled because no vehicle was detected within 30 minutes.")
        st.info("💡 You can make a new reservation if spots are still available.")

    except Exception as e:
        st.error("Error cancelling reservation. Please contact support.")


def _handle_active_status(reservation):
    """Handle active reservations showing remaining time (fixed 15 minutes)."""
    try:
        start_time = pd.to_datetime(reservation.get('start_time'))

        if pd.isna(start_time):
            st.warning("⚠️ Reservation start time is not properly set.")
            return

        now = datetime.now()
        # Fixed 15-minute duration
        fixed_duration_minutes = 15
        end_time = start_time + pd.Timedelta(minutes=fixed_duration_minutes)

        total_duration = fixed_duration_minutes * 60  # Convert to seconds
        elapsed_time = (now - start_time).total_seconds()
        remaining_time = (end_time - now).total_seconds()

        if remaining_time <= 0:
            st.error(" **Reservation Time Expired**")
            st.markdown("Please move your vehicle to avoid penalties.")
        else:
            # Progress bar
            progress = min(1.0, elapsed_time / total_duration) if total_duration > 0 else 0
            st.progress(progress)

            # Time display
            remaining_mins = int(remaining_time // 60)
            remaining_secs = int(remaining_time % 60)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(" Time Remaining", f"{remaining_mins}m {remaining_secs}s")
            with col2:
                st.metric(" Started At", start_time.strftime("%H:%M"))
            with col3:
                st.metric(" Ends At", end_time.strftime("%H:%M"))

            # Warning if time is running low
            if remaining_time < 300:  # Less than 5 minutes
                st.warning("⚠️ **Less than 5 minutes remaining!** Please prepare to move your vehicle.")

            # Auto-refresh every 10 seconds for active reservations
            time.sleep(10)
            st.rerun()

    except Exception as e:
        st.error("Error calculating reservation time.")
        st.exception(e) if st.session_state.get('debug_mode') else None
def _handle_cancelled_status(reservation):
    """Handle cancelled reservations."""
    st.error("❌ **This reservation was cancelled**")

    cancel_reason = reservation.get('cancel_reason', 'Unknown')
    if cancel_reason != 'Unknown':
        st.markdown(f"**Reason:** {cancel_reason}")

    st.info("💡 You can make a new reservation if parking spots are available.")


def _handle_expired_status(reservation):
    """Handle expired reservations."""
    st.warning("⏱️ **Reservation Expired**")

    if 'end_time' in reservation and pd.notna(reservation['end_time']):
        end_time = pd.to_datetime(reservation['end_time'])
        st.caption(f"Expired at: {end_time.strftime('%Y-%m-%d %H:%M')}")

    st.info("💡 You can make a new reservation if parking spots are available.")


def _handle_completed_status(reservation):
    """Handle completed reservations."""
    st.success(" **Reservation Completed Successfully**")

    if 'actual_end_time' in reservation and pd.notna(reservation['actual_end_time']):
        end_time = pd.to_datetime(reservation['actual_end_time'])
        st.caption(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M')}")

    st.balloons()


def _display_reservation_history(reservations):
    """Display a summary of previous reservations."""
    with st.expander(f"📜 Reservation History ({len(reservations)} total)"):
        history_df = reservations.sort_values("created_at", ascending=False)

        for idx, row in history_df.iterrows():
            created = pd.to_datetime(row['created_at']).strftime('%Y-%m-%d %H:%M')
            status_emoji = {
                'completed': '✅',
                'cancelled': '❌',
                'expired': '⏱️',
                'active': '🟢',
                'waiting_detection': '🟡'
            }.get(row['status'].lower(), '⚪')

            st.caption(f"{status_emoji} {created} - Spot {row['spot_id']} - {row['status'].title()}")


def _check_anpr_detection(db, plate_number):
    """Check if ANPR has detected the given plate number recently."""
    try:
        if not os.path.exists(db.anpr_detections_file):
            return False

        detections_df = pd.read_csv(db.anpr_detections_file)
        if detections_df.empty:
            return False

        # Look for recent detections (within last 5 minutes) of this plate
        recent_time = datetime.now() - pd.Timedelta(minutes=5)
        detections_df['detection_time'] = pd.to_datetime(detections_df['detection_time'])

        recent_detections = detections_df[
            (detections_df['plate_number'].str.upper() == plate_number.upper()) &
            (detections_df['detection_time'] >= recent_time)
            ]

        return not recent_detections.empty

    except Exception as e:
        st.error(f"Error checking ANPR detections: {e}")
        return False



def render_copilot_page(db):
    st.header("🤖 Your Friendly Parking Assistant")

    # Friendly welcome message
    st.markdown("#### 👋 Hey there! I'm here to help you find the perfect parking spot!")

    # Add live metrics in a friendly way
    col1, col2, col3 = st.columns(3)

    try:
        spots = db.get_parking_spots()
        available_count = len(spots[spots["status"] == "available"])
        occupied_count = len(spots[spots["status"] == "occupied"])
        total_count = len(spots)

        with col1:
            st.metric("🟢 Ready to Use", available_count, delta=None)
        with col2:
            st.metric("🚗 Currently Parked", occupied_count, delta=None)
        with col3:
            utilization = round((occupied_count / total_count) * 100) if total_count > 0 else 0
            st.metric("📊 How Busy", f"{utilization}%", delta=None)

    except Exception:
        st.info("🔄 Getting the latest parking info for you...")

    st.markdown("---")

    # Friendly suggested questions
    st.markdown("### 💬 Popular Questions")
    st.markdown("*Click any question below or just type naturally!*")

    # User-friendly question categories
    question_groups = {
        " **Quick Checks**": [
            "How many spots are free?",
            "Is parking busy right now?"
        ],
        " **Find the Best Spot**": [
            "Where should I park?",
            "What's the best time to come?"
        ]
    }

    # Create friendly tabs
    tabs = st.tabs(list(question_groups.keys()))

    for i, (category, questions) in enumerate(question_groups.items()):
        with tabs[i]:
            cols = st.columns(len(questions))
            for j, q in enumerate(questions):
                with cols[j]:
                    if st.button(q, key=f"btn_{i}_{j}", help="Click to ask this question"):
                        st.chat_message("user").write(q)
                        response = generate_copilot_response(q, db)
                        st.chat_message("assistant").write(response)

    st.markdown("---")

    # Friendly chat interface
    st.markdown("### 💭 Ask Me Anything!")
    user_input = st.chat_input("Type your question here... I speak human! 😊")

    if user_input:
        st.chat_message("user").write(user_input)
        with st.spinner("Let me check that for you... 🔍"):
            response = generate_copilot_response(user_input, db)
            st.chat_message("assistant").write(response)


def generate_copilot_response(user_input: str, db) -> str:
    """Friendly AI parking assistant that speaks like a helpful human"""

    # Handle empty input with personality
    if not user_input or len(user_input.strip()) < 2:
        return "👋 Hi there! I'm your parking buddy! Ask me anything like:\n• 'How many spots are free?'\n• 'Where should I park?'\n• 'Is it busy right now?'\n\nI'm here to make parking easy for you! 😊"

    user_lower = user_input.lower()

    # Enhanced friendly keyword matching
    def matches_keywords(keywords, threshold=0.6):
        """Smart keyword matching that understands context"""
        for keyword in keywords:
            # Direct match
            if keyword in user_lower:
                return True

            # Fuzzy matching for typos
            if SequenceMatcher(None, keyword, user_lower).ratio() >= threshold:
                return True

            # Smart synonym matching
            synonyms = {
                'available': ['free', 'open', 'vacant', 'empty', 'unused', 'ready'],
                'occupied': ['taken', 'used', 'filled', 'parked', 'busy'],
                'reserved': ['booked', 'saved', 'held', 'scheduled'],
                'busiest': ['crowded', 'popular', 'busy', 'packed', 'full'],
                'zone': ['area', 'section', 'place', 'spot', 'location', 'region'],
                'time': ['when', 'hour', 'period', 'moment'],
                'best': ['good', 'perfect', 'ideal', 'recommended'],
                'current': ['now', 'today', 'right now', 'currently']
            }

            keyword_words = set(keyword.split())
            input_words = set(user_lower.split())

            # Check for word matches including synonyms
            for word in keyword_words:
                if word in input_words:
                    return True
                if word in synonyms:
                    if any(syn in input_words for syn in synonyms[word]):
                        return True

            # Partial matching for complex queries
            matches = sum(1 for kw in keyword_words
                          if any(SequenceMatcher(None, kw, iw).ratio() >= 0.8 for iw in input_words))
            if matches >= max(1, len(keyword_words) * 0.6):
                return True

        return False

    def get_friendly_status_overview(spots_df):
        """Create a friendly, conversational status overview"""
        try:
            status_counts = spots_df["status"].value_counts()
            total = len(spots_df)
            available = status_counts.get("available", 0)
            occupied = status_counts.get("occupied", 0)
            reserved = status_counts.get("reserved", 0)

            # Calculate how busy it is
            busy_spots = occupied + reserved
            utilization = (busy_spots / total * 100) if total > 0 else 0

            # Friendly status description
            if utilization <= 30:
                status_desc = "Nice and quiet! "
                emoji = "🟢"
            elif utilization <= 60:
                status_desc = "Moderately busy "
                emoji = "🟡"
            elif utilization <= 80:
                status_desc = "Getting busy! "
                emoji = "🟠"
            else:
                status_desc = "Very busy right now! "
                emoji = "🔴"

            return f"{emoji} **{status_desc}**\n\n🟢 **{available}** spots ready for you\n🚗 **{occupied}** cars parked\n📅 **{reserved}** spots reserved\n\n💡 **My tip**: {utilization:.0f}% full - {'Perfect time to visit!' if utilization < 50 else 'You might want to hurry!' if utilization < 80 else 'Consider coming back later!'}"

        except Exception:
            return " Having trouble checking the status right now, but I'm still here to help!"

    def get_zone_insights(spots_df, reservations_df):
        """Friendly zone recommendations"""
        try:
            if reservations_df.empty:
                return " I don't have enough history to recommend zones yet, but any available spot should work great!"

            reservations_df = reservations_df.copy()
            reservations_df["zone"] = reservations_df["spot_id"].astype(str).str[0]

            # Find the busiest and quietest zones
            zone_activity = reservations_df["zone"].value_counts()

            if len(zone_activity) > 0:
                busiest_zone = zone_activity.index[0]
                busiest_count = zone_activity.iloc[0]

                quietest_zone = zone_activity.index[-1] if len(zone_activity) > 1 else None
                quietest_count = zone_activity.iloc[-1] if len(zone_activity) > 1 else 0

                result = f" **Zone Insights**\n\n"
                result += f" **Most Popular**: Zone {busiest_zone} ({busiest_count} visits)\n"

                if quietest_zone and quietest_zone != busiest_zone:
                    result += f" **Quieter Option**: Zone {quietest_zone} ({quietest_count} visits)\n"

                result += f"\n💡 **My recommendation**: Try Zone {quietest_zone if quietest_zone else busiest_zone} for a {'peaceful' if quietest_zone else 'convenient'} parking experience!"

                return result

            return "🎯 All zones look equally good right now! Pick any spot that's convenient for you."

        except Exception:
            return " Can't analyze zones right now, but any available spot should work perfectly!"

    def get_time_recommendations(reservations_df):
        """Friendly time-based advice"""
        try:
            if reservations_df.empty:
                return " I don't have enough timing data yet, but right now looks like a great time to park!"

            reservations_df = reservations_df.copy()
            reservations_df["start_time"] = pd.to_datetime(reservations_df["start_time"], errors='coerce')
            reservations_df = reservations_df.dropna(subset=['start_time'])

            if reservations_df.empty:
                return " Right now seems like a perfect time to find a spot!"

            # Analyze by hour
            reservations_df["hour"] = reservations_df["start_time"].dt.hour
            hour_activity = reservations_df["hour"].value_counts().sort_index()

            if not hour_activity.empty:
                # Find peak and quiet hours
                busiest_hour = hour_activity.idxmax()
                busiest_count = hour_activity.max()

                quietest_hours = hour_activity.nsmallest(3).index.tolist()

                current_hour = datetime.now().hour

                result = f" **Perfect Timing Tips**\n\n"
                result += f" **Rush Hour**: Around {busiest_hour}:00 ({busiest_count} people usually park)\n"
                result += f" **Quiet Times**: {', '.join([f'{h}:00' for h in quietest_hours[:2]])}\n\n"

                # Current time advice
                if current_hour == busiest_hour:
                    result += f" **Right Now**: It's peak time! You might need to look around a bit."
                elif current_hour in quietest_hours:
                    result += f" **Right Now**: Perfect timing! Should be easy to find a spot."
                else:
                    result += f" **Right Now**: Pretty good time to park!"

                return result

        except Exception:
            return " Any time is a good time to park! Right now looks perfect! "

    try:
        spots = db.get_parking_spots()
        reservations = db.get_reservations_history()

        # FRIENDLY AVAILABILITY QUERIES
        if re.search(r'\b(how many|number of|count)\b.*\b(available|free|open|spots|empty)\b',
                     user_lower) or matches_keywords(['available spots', 'free spots', 'open spots']):
            available_count = len(spots[spots["status"] == "available"])
            total_count = len(spots)

            if available_count == 0:
                return " **Oops!** No spots available right now, but don't worry - things change quickly! Try checking back in a few minutes."
            elif available_count == 1:
                return f" **Lucky you!** There's **1 spot** waiting for you out of {total_count} total spots!"
            elif available_count <= 5:
                return f"⚡ **Quick!** Only **{available_count} spots** left out of {total_count}. Better grab one fast!"
            else:
                percentage = (available_count / total_count * 100) if total_count > 0 else 0
                return f" **Great news!** **{available_count} spots** are ready for you! That's {percentage:.0f}% of all spots - plenty of choice!"

        # FRIENDLY CURRENT STATUS
        elif matches_keywords(
                ['how is parking', 'parking status', 'current situation', 'busy', 'what\'s it like', 'how busy']):
            return get_friendly_status_overview(spots)

        # FRIENDLY ZONE RECOMMENDATIONS
        elif matches_keywords(
                ['which zone', 'where should I park', 'best area', 'recommend zone', 'quieter zone', 'less busy']):
            return get_zone_insights(spots, reservations)

        # FRIENDLY TIME ADVICE
        elif matches_keywords(['best time', 'when to park', 'good time', 'busy time', 'quiet time', 'rush hour']):
            return get_time_recommendations(reservations)

        # TOTAL SPOTS QUERY
        elif matches_keywords(['total spots', 'how many spots', 'all spots', 'total parking']):
            total = len(spots)
            return f"🏢 **Our parking area** has **{total} spots** total! Pretty nice setup, right? "

        # OCCUPIED/TAKEN SPOTS
        elif matches_keywords(['occupied', 'taken', 'parked', 'used spots']):
            occupied = len(spots[spots["status"] == "occupied"])
            return f" **{occupied} spots** currently have cars parked. Everyone's finding their perfect spot!"

        # RESERVED SPOTS
        elif matches_keywords(['reserved', 'booked', 'scheduled']):
            reserved = len(spots[spots["status"] == "reserved"])
            if reserved == 0:
                return "📅 **No reserved spots** right now - all available spots are yours for the taking!"
            else:
                return f"📅 **{reserved} spots** are reserved for later. Smart planning by those folks!"

        # FRIENDLY HELP
        elif matches_keywords(['help', 'what can you do', 'how does this work', 'commands']):
            return """ **I'm here to make parking super easy for you!**

**Just ask me naturally, like:**
• "How many spots are free?"
• "Is parking busy right now?"
• "Where should I park?"
• "When's the best time to come?"

**I understand casual language, so feel free to ask:**
• "What's the parking situation?"
• "Any good spots available?"
• "Is it crazy busy today?"

**I'm like your friendly parking buddy - just ask me anything! **"""

        # GREETINGS AND PLEASANTRIES
        elif matches_keywords(['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'thanks', 'thank you']):
            greetings = [
                "Hey there!  Ready to find you the perfect parking spot!",
                "Hello!  I'm excited to help you with parking today!",
                "Hi!  Let's get you parked quickly and easily!",
                "Hey!  What can I help you find in our parking area?"
            ]
            return np.random.choice(greetings)

        # SMART FALLBACK WITH HELPFUL SUGGESTIONS
        else:
            # Try to understand what they might want
            if any(word in user_lower for word in ['spot', 'park', 'space']):
                available = len(spots[spots["status"] == "available"])
                return f" **I want to help!** Not sure exactly what you're asking, but I can tell you we have **{available} spots** ready right now!\n\n💡 **Try asking**: 'How many spots are free?' or 'Where should I park?'"

            elif any(word in user_lower for word in ['time', 'when', 'hour']):
                current_hour = datetime.now().hour
                return f" **Time-related question?** Right now at {current_hour}:00 looks like a good time!\n\n💡 **Try asking**: 'When's the best time to park?' or 'Is it busy right now?'"

            else:
                return """ **I'd love to help, but I'm not sure what you're asking!**

**Here's what I'm great at:**
• Telling you how many spots are available
• Showing you the current parking situation  
• Recommending the best areas and times
• Being your friendly parking assistant!

**Just ask me something like:**
• "How's parking today?"
• "Any spots free?"
• "Where should I park?"

*I speak human, so don't worry about being formal! 🤗*"""

    except Exception as e:
        # Friendly error handling
        error_responses = [
            "😅 **Oops!** Something got mixed up on my end. Mind trying that again?",
            "🤔 **Hmm...** I hit a little snag. Could you ask me again?",
            "😊 **Sorry about that!** Something went wonky. Let's try once more!"
        ]
        return np.random.choice(error_responses)


# Admin Spot Grid View with Manual Override
def render_admin_spot_grid(spots_df, db):
    st.header("🗺️ Live Spot Map - Admin Control")
    zones = spots_df['zone'].unique()
    zone_titles = {"B": "ZONE B", "A": "ZONE A", "S": "ZONE S", "E": "ZONE E"}

    for zone in sorted(zones):
        st.subheader(f"Zone {zone} - {zone_titles.get(zone, '')}")
        zone_spots = spots_df[spots_df['zone'] == zone].sort_values('spot_id')
        cols = st.columns(5)
        for idx, (_, spot) in enumerate(zone_spots.iterrows()):
            with cols[idx % 5]:
                color = {
                    'available': '🟢',
                    'reserved': '🟡',
                    'occupied': '🔴',
                    'maintenance': '🔧'
                }.get(spot['status'], '⚪')
                label = f"{color} {spot['spot_id']}"
                st.markdown(label)

                if st.button(f"⚙️ Manage {spot['spot_id']}", key=f"{zone}-{spot['spot_id']}"):
                    with st.form(f"form-{spot['spot_id']}"):
                        new_status = st.selectbox("New Status", ["available", "reserved", "occupied", "maintenance"],
                                                  index=["available", "reserved", "occupied", "maintenance"].index(
                                                      spot['status']))
                        new_plate = st.text_input("Plate Number", spot['plate_number'])
                        reserved_by = st.text_input("Reserved By", spot['reserved_by'])
                        reserved_until = st.text_input("Reserved Until", spot['reserved_until'])
                        if st.form_submit_button("✅ Apply Changes"):
                            db.update_spot_status(spot['spot_id'], new_status, new_plate, reserved_by, reserved_until)
                            st.success(f"Updated {spot['spot_id']}")
                            st.rerun()
    st.markdown("---")
    st.subheader("📊 Spot Map Chart")

    # Reload after updates
    spots_df = db.get_parking_spots()

    spots_df['x'] = spots_df['spot_id'].str.extract('(\d+)').astype(int)
    spots_df['y'] = spots_df['zone'].map({'A': 4, 'B': 3, 'S': 2, 'E': 1})

    # Make sure all statuses appear (even if count = 0)
    for status in ["available", "reserved", "occupied", "maintenance"]:
        if status not in spots_df['status'].unique():
            dummy_row = {
                'spot_id': f'DUMMY_{status}',
                'zone': 'Z',
                'status': status,
                'plate_number': '',
                'reserved_by': '',
                'reserved_until': '',
                'last_updated': '',
                'x': -1,  # hidden from view
                'y': -1
            }
            spots_df = pd.concat([spots_df, pd.DataFrame([dummy_row])], ignore_index=True)

    color_map = {
        "available": "green",
        "reserved": "orange",
        "occupied": "red",
        "maintenance": "gray"
    }

    fig = px.scatter(
        spots_df,
        x='x',
        y='y',
        text='spot_id',
        color='status',
        color_discrete_map=color_map,
        size=[20] * len(spots_df),
        size_max=20,
        height=400
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(
        showlegend=True,
        yaxis_title="Zone",
        xaxis_title="Spot #",
        legend_title="Status"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_user_admin_panel():
    """Enhanced user administration panel with full CRUD operations"""
    from user import UserDatabase
    import streamlit as st
    import pandas as pd
    from datetime import datetime

    st.subheader("👥 User Account Management")

    db = UserDatabase()
    users_df = db.load_users()

    # User Statistics Dashboard
    if not users_df.empty:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("👤 Total Users", len(users_df))



        with col2:
            admin_count = len(users_df[users_df.get('role', 'user') == 'admin']) if 'role' in users_df.columns else 0
            st.metric("🔑 Admins", admin_count)

    # Action Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["️ View Users", " Add User", " Edit User", " Delete User"])

    # TAB 1: View Users
    with tab1:
        st.write("**Current Users:**")

        if not users_df.empty:
            # Search and filter options
            col1, col2 = st.columns([2, 1])

            with col1:
                search_term = st.text_input("🔍 Search users", placeholder="Search by username, email, or phone...")

            with col2:
                if 'role' in users_df.columns:
                    role_filter = st.selectbox("Filter by Role", ["All"] + list(users_df['role'].unique()))
                else:
                    role_filter = "All"

            # Apply filters
            filtered_df = users_df.copy()

            if search_term:
                mask = (
                        filtered_df['username'].str.contains(search_term, case=False, na=False) |
                        filtered_df.get('email', pd.Series([''] * len(filtered_df))).str.contains(search_term,
                                                                                                  case=False,
                                                                                                  na=False) |
                        filtered_df.get('phone', pd.Series([''] * len(filtered_df))).str.contains(search_term,
                                                                                                  case=False, na=False)
                )
                filtered_df = filtered_df[mask]

            if role_filter != "All" and 'role' in users_df.columns:
                filtered_df = filtered_df[filtered_df['role'] == role_filter]

            # Display filtered results
            if not filtered_df.empty:
                # Hide sensitive columns for display
                display_columns = [col for col in filtered_df.columns if col not in ['password_hash', 'password']]
                st.dataframe(filtered_df[display_columns], use_container_width=True)

                # Export option
                if st.button("📥 Export User List (CSV)"):
                    csv_data = filtered_df[display_columns].to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"users_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No users found matching your criteria.")
        else:
            st.info("No users registered yet.")

    # TAB 2: Add User
    with tab2:
        st.write("**Create New User Account:**")

        with st.form("add_user_form"):
            col1, col2 = st.columns(2)

            with col1:
                new_username = st.text_input("👤 Username*", help="Must be unique")
                new_email = st.text_input("📧 Email", help="Optional but recommended")
                new_phone = st.text_input("📱 Phone", help="Optional")

            with col2:
                new_password = st.text_input("🔒 Password*", type="password")
                new_status = st.selectbox("📊 Status", ["active", "inactive", "suspended"])

            new_full_name = st.text_input("📝 Full Name", help="Optional display name")

            submitted = st.form_submit_button("➕ Create User Account")

            if submitted:
                if not new_username or not new_password:
                    st.error("❌ Username and password are required!")
                elif new_username in users_df['username'].values:
                    st.error("❌ Username already exists! Please choose a different one.")
                else:
                    try:
                        # Try using the database's add_user method first
                        if hasattr(db, 'add_user'):
                            success = db.add_user(
                                username=new_username,
                                password=new_password,
                                email=new_email if new_email else None,
                                phone=new_phone if new_phone else None,
                                full_name=new_full_name if new_full_name else None,
                                status=new_status
                            )
                            if success:
                                st.success(f"✅ User '{new_username}' created successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to create user. Please try again.")
                        else:
                            # Fallback: Direct CSV manipulation
                            current_users = db.load_users()

                            # Create new user data
                            import hashlib
                            from datetime import datetime

                            # Determine CSV file path
                            users_file = getattr(db, 'users_file', 'parking_data/users.csv')

                            new_user_data = {
                                'username': new_username,
                                'password_hash': hashlib.sha256(new_password.encode()).hexdigest(),
                                'email': new_email if new_email else '',
                                'phone': new_phone if new_phone else '',
                                'full_name': new_full_name if new_full_name else '',
                                'status': new_status,
                                'created_at': datetime.now().isoformat()
                            }

                            # Add to dataframe
                            new_row = pd.DataFrame([new_user_data])
                            updated_users = pd.concat([current_users, new_row], ignore_index=True)

                            # Save back to CSV
                            updated_users.to_csv(users_file, index=False)

                            st.success(f"✅ User '{new_username}' created successfully!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error creating user: {str(e)}")

    # TAB 3: Edit User
    with tab3:
        st.write("**Modify Existing User:**")

        if not users_df.empty:
            selected_user = st.selectbox("Select User to Edit", users_df['username'].tolist())

            if selected_user:
                user_data = users_df[users_df['username'] == selected_user].iloc[0]

                with st.form("edit_user_form"):
                    col1, col2 = st.columns(2)

                    with col1:
                        edit_email = st.text_input("📧 Email", value=user_data.get('email', ''))
                        edit_phone = st.text_input("📱 Phone", value=user_data.get('phone', ''))
                        edit_full_name = st.text_input("📝 Full Name", value=user_data.get('full_name', ''))

                    with col2:
                        edit_status = st.selectbox("📊 Status", ["active", "inactive", "suspended"],
                                                   index=0 if user_data.get('status', 'active') == 'active' else
                                                   (1 if user_data.get('status', 'active') == 'inactive' else 2))

                    # Password reset option
                    reset_password = st.checkbox("🔄 Reset Password")
                    new_password = ""
                    if reset_password:
                        new_password = st.text_input("🔒 New Password", type="password")

                    submitted = st.form_submit_button("💾 Update User")

                    if submitted:
                        try:
                            # Load current users and update the specific user
                            current_users = db.load_users()
                            user_index = current_users[current_users['username'] == selected_user].index[0]

                            # Update user data
                            if edit_email:
                                current_users.loc[user_index, 'email'] = edit_email
                            if edit_phone:
                                current_users.loc[user_index, 'phone'] = edit_phone
                            if edit_full_name:
                                current_users.loc[user_index, 'full_name'] = edit_full_name

                            current_users.loc[user_index, 'status'] = edit_status

                            # Handle password reset
                            if reset_password and new_password:
                                import hashlib
                                current_users.loc[user_index, 'password_hash'] = hashlib.sha256(
                                    new_password.encode()).hexdigest()

                            # Save back to CSV
                            current_users.to_csv(db.users_file, index=False)

                            st.success(f"✅ User '{selected_user}' updated successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error updating user: {str(e)}")
        else:
            st.info("No users available to edit.")

    # TAB 4: Delete User
    with tab4:
        st.write("**⚠️ Remove User Account:**")
        st.warning("This action cannot be undone!")

        if not users_df.empty:
            delete_user = st.selectbox("Select User to Delete", [""] + users_df['username'].tolist())

            if delete_user:
                user_info = users_df[users_df['username'] == delete_user].iloc[0]

                st.write("**User Details:**")
                st.write(f"👤 Username: {delete_user}")
                st.write(f"📧 Email: {user_info.get('email', 'Not provided')}")
                st.write(f"📊 Status: {user_info.get('status', 'active')}")

                # Confirmation steps
                confirm_text = st.text_input(f"Type '{delete_user}' to confirm deletion:")

                if confirm_text == delete_user:
                    if st.button("🗑️ PERMANENTLY DELETE USER", type="primary"):
                        try:
                            # Try using the database's delete_user method first
                            if hasattr(db, 'delete_user'):
                                success = db.delete_user(delete_user)
                                if success:
                                    st.success(f"✅ User '{delete_user}' has been permanently deleted!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to delete user. Please try again.")
                            else:
                                # Fallback: Direct CSV manipulation
                                current_users = db.load_users()
                                updated_users = current_users[current_users['username'] != delete_user]

                                # Determine CSV file path
                                users_file = getattr(db, 'users_file', 'parking_data/users.csv')

                                # Save the updated dataframe back to CSV
                                updated_users.to_csv(users_file, index=False)

                                st.success(f"✅ User '{delete_user}' has been permanently deleted!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error deleting user: {str(e)}")
                else:
                    st.info("👆 Type the username exactly to enable deletion.")
        else:
            st.info("No users available to delete.")

    # Refresh Button
    st.markdown("---")
    if st.button("🔄 Refresh User Data"):
        st.rerun()




def start_expired_reservations_cleaner(db, interval=60):
    """Run clean_expired_reservations every 'interval' seconds in the background."""
    def cleaner():
        while True:
            try:
                db.clean_expired_reservations()
            except Exception as e:
                print(f"[CLEANER ERROR] {e}")
            time.sleep(interval)

    t = threading.Thread(target=cleaner, daemon=True)
    t.start()

# App entrypoint
def main():
    init_session()
    db = get_db()
    anpr_integration = get_anpr()
    db.clean_expired_reservations()

    st.sidebar.title("🧭 SmartPark Navigation")
    pages = [
        "🏠 Dashboard",
        "🎫 Reservation",
        "🤖 BASE Copilot",
        "📟 Track Status",
        "👤 User Portal",
        "🔐 Admin Login"
    ]

    if st.session_state.admin_logged_in:
        pages += [
            "🎥 ANPR Control",
            "📊 Analytics",
            "🔧 System Settings",
            "🗺️ Admin Spot Map",
            "👥 Manage Users"
        ]

    selection = st.sidebar.radio("Choose a page", pages)

    # 🔄 Always get fresh data
    spots_df = db.get_parking_spots()
    reservations_df = db.get_reservations_history()



    if selection == "🏠 Dashboard":
        render_dashboard_page(db)

    elif selection == "🎫 Reservation":
        render_enhanced_reservation_page(spots_df, db,anpr_integration)

    elif selection == "🎥 ANPR Control":
        render_anpr_dashboard(db, anpr_integration)

    elif selection == "📟 Track Status":
        render_reservation_status_page(db)

    elif selection == "🔐 Admin Login":
        render_admin_login_page()

    elif selection == "👤 User Portal":
        render_user_login_page()

    elif selection == "📊 Analytics":
        render_analytics_page(spots_df, reservations_df)

    elif selection == "🔧 System Settings":
        render_system_settings_page(db)

    elif selection == "🗺️ Admin Spot Map":
        render_admin_spot_map(spots_df, db)

    elif selection == "👥 Manage Users":
        if st.session_state.get("admin_logged_in", False):
            render_user_admin_panel()
        else:
            st.warning("Admins only")

    elif selection == "🤖 BASE Copilot":
        render_copilot_page(db)


if __name__ == "__main__":

    db = ParkingDatabase()
    anpr_integration = ANPRParkingIntegration(db)
    start_expired_reservations_cleaner(db, interval=30)

    main()
