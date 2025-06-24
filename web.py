#web.py
#BuiltWithLove by @papitx0
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import hashlib
import os
from smart_recommender import recommend_best_spot
import re
from notifier import notify_user
import threading
import time
from integrated_anpr_parking import ANPRSystem
import cv2
from admin import check_system_status, render_system_maintenance_message
import random
import string



# ==== Enhanced Database Class ====
class ParkingDatabase:
    def __init__(self, data_dir="parking_data"):
        self.data_dir = data_dir
        self.parking_spots_file = os.path.join(data_dir, "parking_spots.csv")
        self.reservations_file = os.path.join(data_dir, "reservations_history.csv")
        self.emergency_vehicles_file = os.path.join(data_dir, "emergency_vehicles.csv")
        self.admin_users_file = os.path.join(data_dir, "admin_users.csv")
        self.anpr_detections_file = os.path.join(data_dir, "anpr_detections.csv")
        self.queue_file = os.path.join(data_dir, "priority_queue.csv")
        self.init_database()

    def init_database(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.parking_spots_file):
            self.initialize_parking_spots()
        if not os.path.exists(self.reservations_file):
            pd.DataFrame(columns=[
                'id', 'spot_id', 'plate_number', 'customer_name',
                'start_time', 'end_time', 'duration_minutes',
                'detection_time', 'status', 'created_at'
            ]).to_csv(self.reservations_file, index=False)
        if not os.path.exists(self.emergency_vehicles_file):
            pd.DataFrame([{
                "plate_number": "AMB001", "vehicle_type": "Ambulance",
                "description": "City Hospital", "is_active": True,
                "added_date": datetime.now().isoformat()
            }]).to_csv(self.emergency_vehicles_file, index=False)
        if not os.path.exists(self.admin_users_file):
            hash_ = hashlib.sha256("admin123".encode()).hexdigest()
            pd.DataFrame([{
                "username": "admin", "password_hash": hash_, "email": "admin@smartpark.com",
                "role": "super_admin", "created_at": datetime.now().isoformat(), "last_login": ""
            }]).to_csv(self.admin_users_file, index=False)
        if not os.path.exists(self.anpr_detections_file):
            pd.DataFrame(columns=[
                'id', 'plate_number', 'confidence', 'detection_time',
                'camera_location', 'is_emergency', 'processed'
            ]).to_csv(self.anpr_detections_file, index=False)
        if not os.path.exists(self.queue_file):
            pd.DataFrame(columns=["id", "plate_number", "name", "contact", "timestamp", "notified"]) \
                .to_csv(self.queue_file, index=False)

    def initialize_parking_spots(self):
        zones = {"B": "ZONE B", "A": "ZONE A", "S": "ZONE S", "E": "ZONE E"}
        spots = []
        n_spots = 10
        for zone in zones:
            for i in range(1, n_spots + 1):
                spot_id = f"{zone}{i:02}"
                spots.append({
                    "spot_id": spot_id, "zone": zone, "status": "available",
                    "plate_number": "", "reserved_by": "", "reserved_until": "",
                    "last_updated": datetime.now().isoformat()
                })
        pd.DataFrame(spots).to_csv(self.parking_spots_file, index=False)

    def get_parking_spots(self):
        return pd.read_csv(self.parking_spots_file)

    def get_reservations_history(self):
        try:
            return pd.read_csv(self.reservations_file)
        except:
            return pd.DataFrame()

    def add_reservation(self, spot_id, plate_number, name, duration, contact=None):
        """Add a reservation and send a verification code if contact is provided."""
        from notifier import notify_user  # Ensure it's always available
        import random
        import string
        from datetime import datetime, timedelta
        import pandas as pd
        import os


        start = datetime.now()

        if duration == "Unlimited":
            end = start + timedelta(days=3650)  # 10 years
            duration_minutes = "Unlimited"
        else:
            end = start + timedelta(minutes=duration)
            duration_minutes = duration

        # Generate a random 6-digit verification code
        verification_code = ''.join(random.choices(string.digits, k=6))

        df = self.get_reservations_history()
        new_id = len(df) + 1

        new_row = pd.DataFrame([{
            "id": new_id,
            "spot_id": spot_id,
            "plate_number": plate_number,
            "customer_name": name,
            "start_time": "",
            "end_time": "",
            "duration_minutes": duration,
            "detection_time": "",
            "status": "waiting_detection",
            "verification_code": verification_code,
            "created_at": datetime.now().isoformat()
        }])

        df = pd.concat([df, new_row])
        df.to_csv(self.reservations_file, index=False)

        # Update spot status
        self.update_spot_status(
            spot_id=spot_id,
            status='reserved',
            plate_number=plate_number,
            reserved_by=name,
            reserved_until=end.isoformat()
        )

        # Optional: Send confirmation and verification code
        if contact:
            message = f"""
     Hello {name},

    Your SmartPark reservation for spot **{spot_id}** has been confirmed!
    Plate Number: **{plate_number}**
    Duration: **{duration_minutes} minutes**

    🎁 **Redeem Code for Reward Points:** `{verification_code}`  
    Keep it safe — you'll be able to redeem points when your plate is detected.

    Thank you for choosing BASE SmartPark — making parking Smart, Secure, and Seamless! 
    """
            try:
                notify_user(contact, message)
            except Exception as e:
                print(f"Warning: Could not send notification — {e}")

        return verification_code


    def update_spot_status(self, spot_id, status, plate_number='', reserved_by='', reserved_until=''):
        df = pd.read_csv(self.parking_spots_file)
        mask = df['spot_id'] == spot_id
        df.loc[mask, 'status'] = status
        df.loc[mask, 'plate_number'] = plate_number
        df.loc[mask, 'reserved_by'] = reserved_by
        df.loc[mask, 'reserved_until'] = reserved_until
        df.loc[mask, 'last_updated'] = datetime.now().isoformat()
        df.to_csv(self.parking_spots_file, index=False)

    def process_anpr_detection(self, plate_number, confidence, is_emergency=False):
        """Process ANPR detection and update parking status"""
        current_time = datetime.now()

        # Generate a random 6-digit code
        verification_code = ''.join(random.choices(string.digits, k=6))

        # Check for existing reservations
        reservations_df = self.get_reservations_history()
        waiting_reservation = reservations_df[
            (reservations_df['plate_number'].str.upper() == plate_number.upper()) &
            (reservations_df['status'] == 'waiting_detection')
            ]

        if not waiting_reservation.empty:
            # Vehicle with reservation detected - activate reservation
            reservation = waiting_reservation.iloc[0]
            reservation_id = reservation['id']
            spot_id = reservation['spot_id']
            duration = reservation['duration_minutes']

            # Calculate end time
            if duration == "Unlimited":
                end_time = current_time + timedelta(days=3650)
            else:
                end_time = current_time + timedelta(minutes=int(duration))

            # Update reservation status
            reservations_df.loc[reservations_df['id'] == reservation_id, 'status'] = 'active'
            reservations_df.loc[reservations_df['id'] == reservation_id, 'start_time'] = current_time.isoformat()
            reservations_df.loc[reservations_df['id'] == reservation_id, 'end_time'] = end_time.isoformat()
            reservations_df.loc[reservations_df['id'] == reservation_id, 'detection_time'] = current_time.isoformat()
            reservations_df.loc[
                reservations_df['id'] == reservation_id, 'verification_code'] = verification_code  # Add this line
            reservations_df.to_csv(self.reservations_file, index=False)

            # Update spot status to occupied
            self.update_spot_status(
                spot_id=spot_id,
                status='occupied',
                plate_number=plate_number,
                reserved_by=reservation['customer_name'],
                reserved_until=end_time.isoformat()
            )

            return {
                'action': 'reservation_activated',
                'spot_id': spot_id,
                'plate_number': plate_number,
                'message': f"Reservation activated for {plate_number} at spot {spot_id}",
                'verification_code': verification_code  # Add this line
            }

        elif is_emergency:
            # Emergency vehicle detected - find nearest available spot
            available_spots = self.get_parking_spots()
            available_spots = available_spots[available_spots['status'] == 'available']

            if not available_spots.empty:
                # Assign first available spot for emergency
                emergency_spot = available_spots.iloc[0]['spot_id']

                # Create emergency reservation
                df = self.get_reservations_history()
                new_id = len(df) + 1
                emergency_end = current_time + timedelta(hours=4)  # 4-hour emergency slot

                new_row = pd.DataFrame([{
                    "id": new_id,
                    "spot_id": emergency_spot,
                    "plate_number": plate_number,
                    "customer_name": "EMERGENCY VEHICLE",
                    "start_time": current_time.isoformat(),
                    "end_time": emergency_end.isoformat(),
                    "duration_minutes": 240,
                    "detection_time": current_time.isoformat(),
                    "status": "active",
                    "created_at": current_time.isoformat()
                }])

                df = pd.concat([df, new_row])
                df.to_csv(self.reservations_file, index=False)

                # Update spot status
                self.update_spot_status(
                    spot_id=emergency_spot,
                    status='occupied',
                    plate_number=plate_number,
                    reserved_by='EMERGENCY VEHICLE',
                    reserved_until=emergency_end.isoformat()
                )

                return {
                    'action': 'emergency_assigned',
                    'spot_id': emergency_spot,
                    'plate_number': plate_number,
                    'message': f"Emergency vehicle {plate_number} assigned to spot {emergency_spot}"
                }

        else:
            # Unknown vehicle detected - add to queue or suggest registration
            return {
                'action': 'unknown_vehicle',
                'plate_number': plate_number,
                'message': f"Unknown vehicle {plate_number} detected. Please make a reservation or join the queue."
            }

    def clean_expired_reservations(self):
        now = datetime.now()
        df = self.get_reservations_history()
        spots_df = self.get_parking_spots()
        updated = False

        for i, row in df.iterrows():
            if row['status'] == 'active' and row['end_time']:
                try:
                    end_time = datetime.fromisoformat(row['end_time'])
                    if end_time < now:
                        # Mark reservation as expired
                        df.at[i, 'status'] = 'expired'

                        # Update spot status to available
                        self.update_spot_status(
                            spot_id=row['spot_id'],
                            status='available',
                            plate_number='',
                            reserved_by='',
                            reserved_until=''
                        )

                        # Notify next person in queue
                        next_user = self.notify_next_user_in_queue()
                        if next_user is not None:
                            message = f"Good news! A parking spot is now available. Please proceed to make a reservation."
                            notify_user(next_user['contact'], message)

                        updated = True
                except:
                    continue

        if updated:
            df.to_csv(self.reservations_file, index=False)

    def get_queue(self):
        return pd.read_csv(self.queue_file)

    def add_to_queue(self, plate, name, contact):
        df = self.get_queue()
        new_id = len(df) + 1
        new_entry = pd.DataFrame([{
            "id": new_id,
            "plate_number": plate,
            "name": name,
            "contact": contact,
            "timestamp": datetime.now().isoformat(),
            "notified": False
        }])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(self.queue_file, index=False)

    def notify_next_user_in_queue(self):
        df = self.get_queue()
        pending = df[df['notified'] == False]
        if not pending.empty:
            first = pending.iloc[0]
            df.loc[pending.index[0], 'notified'] = True
            df.to_csv(self.queue_file, index=False)
            return first
        return None


# ==== ANPR Integration Class ====
class ANPRParkingIntegration:
    def __init__(self, db: ParkingDatabase):
        self.db = db
        self.anpr_system = None
        self.monitoring_active = False
        self.monitoring_thread = None

    def initialize_anpr(self):
        """Initialize ANPR system"""
        try:
            self.anpr_system = ANPRSystem(
                confidence_threshold=0.6,
                ocr_confidence_threshold=0.5
            )
            return True
        except Exception as e:
            st.error(f"Failed to initialize ANPR system: {e}")
            return False

    def start_monitoring(self, camera_id=0):
        """Start continuous ANPR monitoring in background"""
        if not self.anpr_system:
            if not self.initialize_anpr():
                return False

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_camera,
            args=(camera_id,),
            daemon=True
        )
        self.monitoring_thread.start()
        return True

    def stop_monitoring(self):
        """Stop ANPR monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

    def _monitor_camera(self, camera_id):
        """Background camera monitoring function"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame_count = 0
        last_detection_time = {}
        detection_cooldown = 10  # 10 seconds between processing same plate

        try:
            while self.monitoring_active:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                current_time = time.time()

                # Process every 10th frame to reduce load
                if frame_count % 10 == 0:
                    # Save frame temporarily
                    temp_frame_path = "temp_monitoring_frame.jpg"
                    cv2.imwrite(temp_frame_path, frame)

                    # Process frame
                    detections = self.anpr_system.process_image(temp_frame_path)

                    # Process each detection
                    for detection in detections:
                        plate_number = detection['plate_number']

                        # Check cooldown period
                        if (plate_number not in last_detection_time or
                                current_time - last_detection_time[plate_number] > detection_cooldown):
                            # Process the detection
                            result = self.db.process_anpr_detection(
                                plate_number=plate_number,
                                confidence=detection['confidence'],
                                is_emergency=detection['is_emergency']
                            )

                            # Save detection to database
                            self.anpr_system.save_detections([detection])

                            # Update last detection time
                            last_detection_time[plate_number] = current_time

                            # Log the result
                            print(f"ANPR Detection: {result['message']}")

                    # Clean up temp file
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)

                # Clean expired reservations periodically
                if frame_count % 300 == 0:  # Every ~30 seconds at 10 FPS processing
                    self.db.clean_expired_reservations()

                time.sleep(0.1)  # Small delay to prevent excessive CPU usage

        except Exception as e:
            print(f"Error in camera monitoring: {e}")
        finally:
            cap.release()

    def process_single_image(self, image_path):
        """Process a single image for testing"""
        if not self.anpr_system:
            if not self.initialize_anpr():
                return []

        detections = self.anpr_system.process_image(image_path)
        results = []

        for detection in detections:
            result = self.db.process_anpr_detection(
                plate_number=detection['plate_number'],
                confidence=detection['confidence'],
                is_emergency=detection['is_emergency']
            )
            results.append({
                'detection': detection,
                'parking_result': result
            })

        return results


# ==== Enhanced UI Functions ====

def render_enhanced_reservation_page(spots_df, db, anpr_integration):
    # Import UserDatabase to handle verification codes
    from user import UserDatabase

    # Check system status first
    system_status = check_system_status(db)

    if not system_status.get("system_enabled", True):
        render_system_maintenance_message()
        return

    if not system_status.get("reservations_enabled", True):
        st.warning("🚫 **Reservation System Temporarily Disabled**")
        st.write("The reservation system is currently offline for maintenance.")
        st.write("Please check back later or contact support for assistance.")
        return

    st.header("🎫 Make a Reservation")
    available = spots_df[spots_df['status'] == 'available']
    disabled_zones = check_system_status(db).get("disabled_zones", [])
    available = available[~available["zone"].isin(disabled_zones)]

    if disabled_zones:
        st.warning(f"⚠️ Zones {', '.join(disabled_zones)} are currently disabled for reservation.")

    if available.empty:
        st.warning("🚫 No available spots at the moment.")

        st.subheader("📥 Join the Waiting Queue")
        with st.form("queue_form"):
            name = st.text_input("Your Name")
            plate = st.text_input("License Plate Number")
            contact = st.text_input("Contact Info (Email or Phone)")

            submitted = st.form_submit_button("📩 Join Queue")

            if submitted:
                contact = contact.strip()
                is_email = re.fullmatch(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", contact)
                is_phone = re.fullmatch(r"\+?[0-9]{7,15}", contact)

                if not is_email and not is_phone:
                    st.error("❌ Invalid contact format. Use a valid email or phone number.")
                else:
                    contact_type = "email" if is_email else "phone"
                    db.add_to_queue(plate.upper(), name, contact)
                    st.success(f"✅ Added to the queue! You'll be notified via your {contact_type}.")

                    confirm_msg = f""" Hello {name},
You've been added to the SmartPark waiting queue.
You'll receive another alert when a parking spot becomes available.
Your plate: {plate.upper()}"""
                    notify_user(contact, confirm_msg)
        return

    # ANPR Auto-Reservation Mode
    st.subheader(" ANPR Auto-Reservation")
    st.info("💡 This mode will automatically assign a spot when your license plate is detected by our cameras.")

    with st.form("anpr_reservation"):
        col1, col2 = st.columns(2)

        with col1:
            # Auto-generated username
            user_id = random.randint(1000, 9999)
            auto_name = f"user{user_id}"
            st.text_input("Auto-Generated User ID", value=auto_name, disabled=True)

            plate = st.text_input("License Plate Number")

            # Optional username for verification code
            username = st.text_input(
                "Username (optional and important for verification code)",
                help="Enter your username if you want to receive a verification code (you can skip this)."
            )
        with col2:
            duration_options = ["15", "30", "60", "120", "180", "Unlimited"]
            duration_str = st.selectbox("Duration (minutes)", duration_options, index=2)

        # AI Suggestion
        suggested_spot = recommend_best_spot(None, available)
        if suggested_spot:
            st.info(f" AI suggests best available spot: **{suggested_spot}**")
            default_spot = suggested_spot
        else:
            default_spot = available.iloc[0]["spot_id"]

        selected_spot = st.selectbox(
            "Pre-assign Spot",
            available["spot_id"].tolist(),
            index=available["spot_id"].tolist().index(default_spot),
        )
        duration = 525600 if duration_str == "Unlimited" else int(duration_str)

        if st.form_submit_button("🎯 Create ANPR Reservation"):
            if plate.strip() == "":
                st.warning("⚠️ License plate number is required to create a reservation.")
            else:
                # Create the reservation regardless of username
                db.add_reservation(selected_spot, plate.upper(), auto_name, duration)

                if username.strip():
                    # User entered a username, try generating verification code
                    user_db = UserDatabase()
                    try:
                        verification_code = user_db.generate_verification_code(username.strip(),
                                                                               75)  # 75 points for ANPR
                        st.session_state.pending_redeem_code = verification_code
                        st.session_state.reservation_success = "anpr"
                        st.session_state.username_for_code = username.strip()
                        st.success(
                            f"✅ ANPR Reservation created! Spot {selected_spot} will be activated when plate {plate.upper()} is detected."
                        )
                        st.info(
                            "🎥 Drive to the parking area — our cameras will automatically detect your plate and activate your reservation."
                        )
                    except Exception as e:
                        st.error(f"❌ Error creating verification code: {e}")
                        st.session_state.pending_redeem_code = f"ANPR-{random.randint(100000, 999999)}"
                        st.session_state.reservation_success = "anpr"
                        st.success(
                            f"✅ ANPR Reservation created (without user verification). Spot {selected_spot} will be activated when plate {plate.upper()} is detected."
                        )
                else:
                    # No username, skip verification code
                    st.session_state.reservation_success = "anpr"
                    st.success(
                        f"✅ ANPR Reservation created! Spot {selected_spot} will be activated when plate {plate.upper()} is detected."
                    )
                    st.info(
                        "🎥 Drive to the parking area — our cameras will automatically detect your plate and activate your reservation."
                    )

    # ⚡ Demo: Simulate ANPR Detection
    def generate_random_plate():
        letters = ''.join(random.choices(string.ascii_uppercase, k=3))
        numbers = ''.join(random.choices(string.digits, k=4))
        return f"{letters}{numbers}"

    st.subheader("⚡ Demo: Simulate Random ANPR Detection")

    # Initialize session state for demo results if not exists
    if 'last_demo_result' not in st.session_state:
        st.session_state.last_demo_result = None

    # Button to simulate random detection
    if st.button(" Simulate Random Plate Detection", key="simulate_anpr_detection"):
        selected_spot = available.iloc[0]['spot_id']
        detected_plate = generate_random_plate()
        db.add_reservation(
            spot_id=selected_spot,
            plate_number=detected_plate,
            name="DEMO",
            duration=60
        )

        # Store the result in session state
        st.session_state.last_demo_result = {
            'plate': detected_plate,
            'confidence': random.random(),  # Random confidence between 0 and 1
            'result': "Successfully processed"
        }

        # Just create a demo code (no username required)
        verification_code = f"DEMO-{random.randint(100000, 999999)}"
        st.session_state.pending_redeem_code = verification_code
        st.session_state.reservation_success = "demo"

        st.success(" Random plate detected and processed! A demo verification code has been generated.")

    # Display last result if available
    if st.session_state.last_demo_result:
        st.write("### Last Detection Result")
        st.write(f"**Plate:** {st.session_state.last_demo_result['plate']}")
        st.write(f"**Confidence:** {st.session_state.last_demo_result['confidence'] * 100:.2f}%")
        st.write(f"**Result:** {st.session_state.last_demo_result['result']}")

    # ===== END OF DEMO BLOCK =====

    # Manual Reservation (existing functionality)
    st.subheader("📋 Manual Reservation")

    global_best = recommend_best_spot(None, available)
    if global_best:
        st.info(f" AI Global Suggestion: Best overall spot is **{global_best}**")

    zone_list = sorted(available["zone"].unique())
    if "selected_zone_manual" not in st.session_state or st.session_state.selected_zone_manual not in zone_list:
        st.session_state.selected_zone_manual = zone_list[0] if zone_list else None


    zone = st.selectbox("Zone", zone_list, index=zone_list.index(st.session_state.selected_zone_manual))
    st.session_state.selected_zone_manual = zone
    filtered_spots = available[available["zone"] == zone]["spot_id"].tolist()

    suggested = recommend_best_spot(zone, available)

    if suggested:
        st.info(f" AI Suggests: Best spot in Zone {zone} is **{suggested}**")

    with st.form("reserve_manual"):
        if filtered_spots:
            default_idx = filtered_spots.index(suggested) if suggested in filtered_spots else 0
            spot = st.selectbox("Spot", filtered_spots, index=default_idx, key=f"spot_manual_{zone}")
        else:
            st.warning("No available spots in this zone.")
            return
        plate = st.text_input("Plate Number", key="manual_plate")
        duration_options = ["15", "30", "60", "120", "180", "Unlimited"]
        duration_str = st.selectbox("Duration", duration_options, index=2, key="manual_duration")
        st.info("""
                    **Why we need your name and email:**
                    - Name allows us to personalize and address you properly.
                    - Email ensures we can send timely confirmations and alerts.
                    - Your privacy is important. This information is used exclusively for parking-related notifications.
                    - You'll get alerts when your spot is ready, about to expire, or if any changes occur.
                    """)
        name = st.text_input("Name (optional)", key="manual_name")
        email = st.text_input("Email (optional)", key="manual_email")

        # Optional username
        username_manual = st.text_input(
            "Username (optional and important for verification code)",
            help="Enter your username if you want to receive a verification code (you can skip this).",
            key="manual_username"
        )
        duration = 525600 if duration_str == "Unlimited" else int(duration_str)

        if st.form_submit_button("Reserve Now"):
            if plate.strip() == "":
                st.warning("⚠️ License plate is required.")
                return

            # Proceed with the reservation regardless of username
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=duration) if duration != 525600 else start_time + timedelta(
                days=3650)

            df = db.get_reservations_history()
            new_id = len(df) + 1

            new_row = pd.DataFrame([{
                "id": new_id,
                "spot_id": spot,
                "plate_number": plate.upper(),
                "customer_name": name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_minutes": duration,
                "detection_time": start_time.isoformat(),
                "status": "active",
                "created_at": start_time.isoformat()
            }])

            df = pd.concat([df, new_row])
            df.to_csv(db.reservations_file, index=False)

            db.update_spot_status(
                spot_id=spot,
                status="occupied",
                plate_number=plate.upper(),
                reserved_by=name,
                reserved_until=end_time.isoformat()
            )

            # Optional verification code generation
            if username_manual.strip():
                user_db = UserDatabase()
                try:
                    verification_code = user_db.generate_verification_code(username_manual.strip(),
                                                                           100)  # 100 points for manual
                    st.session_state.pending_redeem_code = verification_code
                    st.session_state.reservation_success = "manual"
                    st.session_state.username_for_code = username_manual.strip()
                    st.session_state.reservation_details = {
                        "spot": spot,
                        "plate": plate.upper(),
                        "name": name,
                        "email": email.strip() if email and email.strip() else None,
                        "duration_str": duration_str,
                        "start_time": start_time,
                        "redeem_code": verification_code,
                        "username": username_manual.strip()
                    }
                    st.success("✅ Manual reservation created and activated! Verification code generated.")
                except Exception as e:
                    st.error(f"❌ Error creating verification code: {e}")
                    redeem_code = f"MANUAL-{random.randint(100000, 999999)}"
                    st.session_state.pending_redeem_code = redeem_code
                    st.session_state.reservation_success = "manual"
                    st.success("✅ Manual reservation created and activated (without verification code).")
            else:
                # No username entered, skip verification code
                st.session_state.reservation_success = "manual"
                st.success("✅ Manual reservation created and activated! No verification code required.")

    # ===== REDEEM CODE SECTION - AT THE BOTTOM =====
    handle_redeem_code_display()


def handle_redeem_code_display():
    """Handle redeem code display and processing at the bottom of the page"""

    # Display existing redeem code if it exists
    if "redeem_code" in st.session_state and st.session_state.redeem_code:
        st.success(f"🎁 **Your Redeem Code:** `{st.session_state.redeem_code}`")
        st.info("📝 Save this code! You can use it in the User Portal to earn points.")

        # Show username if available
        if "username_for_code" in st.session_state:
            st.info(f"👤 This code is for username: **{st.session_state.username_for_code}**")

        # Add button to clear the code after user acknowledges it
        if st.button("✅ I've saved my redeem code", key="clear_redeem_code"):
            if "redeem_code" in st.session_state:
                del st.session_state.redeem_code
            if "username_for_code" in st.session_state:
                del st.session_state.username_for_code
            st.rerun()

    # Process pending redeem code from recent reservations
    if "pending_redeem_code" in st.session_state:
        # Set the redeem code to display
        st.session_state.redeem_code = st.session_state.pending_redeem_code

        # Handle email sending for manual reservations
        if (st.session_state.get("reservation_success") == "manual" and
                "reservation_details" in st.session_state):

            details = st.session_state.reservation_details
            if details.get('email'):
                thank_you_msg = f"""
Dear {details['name']},

Thank you for choosing BASE SmartPark. We appreciate your trust in our services and are pleased to confirm your parking reservation.

**Reservation Details:**
- Parking Spot: {details['spot']}
- License Plate: {details['plate']}
- Duration: {details['duration_str']} minutes
- Reservation Start Time: {details['start_time'].strftime('%Y-%m-%d %H:%M:%S')}
- Your Verification Code: {details['redeem_code']}
- Username: {details.get('username', 'N/A')}

**How to Use Your Verification Code:**
1. Go to the User Portal
2. Login with username: {details.get('username', 'your username')}
3. Navigate to "Verify Code" tab
4. Enter your verification code: {details['redeem_code']}
5. Earn bonus points instantly!

Note: Verification codes expire after 15 minutes, so redeem yours soon!

Our team is committed to providing you with a seamless and convenient parking experience. 
Should you have any questions or require further assistance, please don't hesitate to contact us.

Thank you for using BASE SmartPark

Best regards,  
BASE SmartPark Team  
Smart. Secure. Seamless.
                """
                try:
                    notify_user(details['email'], thank_you_msg)
                    st.info(f"📧 Thank you email sent to {details['email']}")
                except:
                    st.warning("⚠️ Could not send email notification")

        # Clean up session state
        del st.session_state.pending_redeem_code
        if "reservation_success" in st.session_state:
            del st.session_state.reservation_success
        if "reservation_details" in st.session_state:
            del st.session_state.reservation_details

        # Rerun to show the redeem code
        st.rerun()



# Modified function for ANPR Dashboard with Recent Activity
def render_anpr_dashboard(db, anpr_integration):
    system_status = check_system_status(db)

    if not system_status.get("system_enabled", True):
        render_system_maintenance_message()
        return

    if not system_status.get("anpr_enabled", True):
        st.warning("🚫 **ANPR System Temporarily Disabled**")
        st.write("The ANPR monitoring system is currently offline for maintenance.")
        st.write("Manual reservations may still be available.")
        return

    st.header("🎥 ANPR Parking Management")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 Live Monitoring")

        if 'monitoring_active' not in st.session_state:
            st.session_state.monitoring_active = False

        if not st.session_state.monitoring_active:
            camera_id = st.selectbox("Select Camera", [0, 1, 2], index=0)
            if st.button("🎬 Start Live Monitoring"):
                if anpr_integration.start_monitoring(camera_id):
                    st.session_state.monitoring_active = True
                    st.success("✅ ANPR monitoring started!")
                    st.rerun()
                else:
                    st.error("❌ Failed to start monitoring")
        else:
            st.success("🔴 ANPR Monitoring Active")
            if st.button("⏹️ Stop Monitoring"):
                anpr_integration.stop_monitoring()
                st.session_state.monitoring_active = False
                st.success("⏸️ Monitoring stopped")
                st.rerun()
    """""""""
    with col2:
        st.subheader("📊 Recent ANPR Detections")

        # Show recent detections
        if os.path.exists(db.anpr_detections_file):
            detections_df = pd.read_csv(db.anpr_detections_file)
            if not detections_df.empty:
                recent_detections = detections_df.tail(5)
                for _, detection in recent_detections.iterrows():
                    with st.container():
                        emergency_icon = "🚨" if detection.get('is_emergency', False) else "🚗"
                        st.write(f"{emergency_icon} **{detection['plate_number']}** "
                                 f"(Confidence: {detection['confidence']:.2f}) "
                                 f"- {detection['detection_time']}")
        else:
            st.info("No detections yet")

    # Recent Activity Section (moved from dashboard)
    st.subheader("📈 Recent Activity")
    reservations_df = db.get_reservations_history()
    if not reservations_df.empty:
        recent = reservations_df.tail(5).sort_values('created_at', ascending=False)
        for _, res in recent.iterrows():
            status_emoji = {
                'active': '🟢',
                'waiting_detection': '🟡',
                'expired': '🔴',
                'cancelled': '❌'
            }.get(res['status'], '⚪')
            st.write(f"{status_emoji} {res['plate_number']} - Spot {res['spot_id']} - {res['status']}")
    else:
        st.info("No recent activity")
"""""""""
"""""""""
    # Test section
    st.subheader("🧪 Test ANPR Detection")
    uploaded_file = st.file_uploader("Upload image for testing", type=['jpg', 'jpeg', 'png'])

    if uploaded_file and st.button("🔍 Analyze Image"):
        # Save uploaded file temporarily
        temp_path = f"temp_upload_{int(time.time())}.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process image
        with st.spinner("Processing image..."):
            results = anpr_integration.process_single_image(temp_path)

        # Display results
        if results:
            st.success(f"✅ Found {len(results)} license plate(s)")
            for i, result in enumerate(results):
                detection = result['detection']
                parking_result = result['parking_result']

                with st.expander(f"Detection {i + 1}: {detection['plate_number']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Detection Details:**")
                        st.write(f"Plate: {detection['plate_number']}")
                        st.write(f"Confidence: {detection['confidence']:.3f}")
                        st.write(f"Emergency: {'Yes' if detection['is_emergency'] else 'No'}")

                    with col2:
                        st.write("**Parking Action:**")
                        st.write(f"Action: {parking_result['action']}")
                        st.write(f"Message: {parking_result['message']}")
                        if 'spot_id' in parking_result:
                            st.write(f"Spot: {parking_result['spot_id']}")
        else:
            st.warning("No license plates detected in the image")

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

"""""""""
plate_read = "integrated_anpr_parking.py"
csv_cleaner = "parking_data/csv_cleaner.py"
'''''''''                                   auto opening

try:
    print(f'Launching {plate_read}')
    print(f'Launching {csv_cleaner}')
    subprocess.run(["python", plate_read], check=True)
    subprocess.run(["python", csv_cleaner], check=True)
except subprocess.CalledProcessError as e:
    print(f'Error \n Verify file existence: {e}')
'''''''''



# Initialize global ANPR integration
@st.cache_resource
def get_anpr_integration():
    db = ParkingDatabase()
    return ANPRParkingIntegration(db)


# Export the enhanced functions
__all__ = [
    "ParkingDatabase",
    "ANPRParkingIntegration",
    "render_anpr_dashboard",
    "render_enhanced_reservation_page",
    "get_anpr_integration"
]