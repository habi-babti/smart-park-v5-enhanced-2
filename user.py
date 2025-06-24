#user.py
#BuiltWithLove by @papitx0
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
from passlib.context import CryptContext
import random
import string

# Password hashing configuration
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    default="pbkdf2_sha256",
    pbkdf2_sha256__default_rounds=30000
)


# === User DB Handler ===
class UserDatabase:
    def __init__(self, data_dir="parking_data"):
        self.users_file = os.path.join(data_dir, "users.csv")
        self.verification_codes_file = os.path.join(data_dir, "verification_codes.csv")
        self.data_dir = data_dir
        self.init_users()
        self.init_verification_codes()

    def init_users(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.users_file):
            df = pd.DataFrame(columns=[
                "username", "password_hash", "points", "created_at", "last_login"
            ])
            df.to_csv(self.users_file, index=False)

    def init_verification_codes(self):
        """Initialize verification codes CSV file"""
        if not os.path.exists(self.verification_codes_file):
            df = pd.DataFrame(columns=[
                "code", "username", "points_reward", "created_at", "expires_at", "used", "used_at"
            ])
            df.to_csv(self.verification_codes_file, index=False)

    def hash_password(self, password):
        return pwd_context.hash(password)

    def verify_password(self, password, hashed):
        return pwd_context.verify(password, hashed)

    def load_users(self):
        return pd.read_csv(self.users_file)

    def save_users(self, df):
        df.to_csv(self.users_file, index=False)

    def load_verification_codes(self):
        """Load verification codes from CSV"""
        return pd.read_csv(self.verification_codes_file)

    def save_verification_codes(self, df):
        """Save verification codes to CSV"""
        df.to_csv(self.verification_codes_file, index=False)

    def generate_verification_code(self, username, points_reward=50):
        """Generate a new verification code for a user"""
        # Generate a random 6-character code
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

        # Make sure code is unique
        df = self.load_verification_codes()
        while code in df['code'].values:
            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

        # Set expiration time to 15 minutes from now
        created_at = datetime.now()
        expires_at = created_at + timedelta(minutes=15)

        # Add new code to dataframe
        new_code = pd.DataFrame([{
            "code": code,
            "username": username,
            "points_reward": points_reward,
            "created_at": created_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "used": False,
            "used_at": ""
        }])

        df = pd.concat([df, new_code], ignore_index=True)
        self.save_verification_codes(df)

        return code

    def verify_redeem_code(self, code, username):
        """Verify and redeem a verification code"""
        df = self.load_verification_codes()

        # Find the code
        code_row = df[df['code'] == code.upper()]

        if code_row.empty:
            return False, "Invalid verification code."

        code_data = code_row.iloc[0]
        code_index = code_row.index[0]

        # Check if code is for this user
        if code_data['username'] != username:
            return False, "This code is not assigned to your account."

        # Check if code is already used
        if code_data['used']:
            return False, "This verification code has already been used."

        # Check if code has expired
        expires_at = datetime.fromisoformat(code_data['expires_at'])
        if datetime.now() > expires_at:
            return False, "This verification code has expired (codes are valid for 15 minutes)."

        # Code is valid - mark as used and award points
        df.at[code_index, 'used'] = True
        df.at[code_index, 'used_at'] = datetime.now().isoformat()
        self.save_verification_codes(df)

        # Award points to user
        points_reward = int(code_data['points_reward'])
        self.add_points(username, points_reward)

        return True, f"Success! You've earned {points_reward} points."

    def get_user_verification_codes(self, username):
        """Get all verification codes for a user (active ones)"""
        df = self.load_verification_codes()
        user_codes = df[df['username'] == username]

        # Filter for active codes (not used and not expired)
        active_codes = []
        current_time = datetime.now()

        for _, code_row in user_codes.iterrows():
            if not code_row['used']:
                expires_at = datetime.fromisoformat(code_row['expires_at'])
                if current_time <= expires_at:
                    time_left = expires_at - current_time
                    minutes_left = int(time_left.total_seconds() / 60)
                    active_codes.append({
                        'code': code_row['code'],
                        'points': code_row['points_reward'],
                        'expires_in_minutes': minutes_left
                    })

        return active_codes

    def cleanup_expired_codes(self):
        """Remove expired codes from the system"""
        df = self.load_verification_codes()
        current_time = datetime.now()

        # Keep only non-expired codes or used codes (for history)
        valid_codes = []
        for _, row in df.iterrows():
            expires_at = datetime.fromisoformat(row['expires_at'])
            if row['used'] or current_time <= expires_at:
                valid_codes.append(row)

        if valid_codes:
            clean_df = pd.DataFrame(valid_codes)
            self.save_verification_codes(clean_df)

    def signup(self, username, password):
        df = self.load_users()
        if username in df['username'].values:
            return False, "Username already exists."
        new_user = pd.DataFrame([{
            "username": username,
            "password_hash": self.hash_password(password),
            "points": 10,  # First login reward
            "created_at": datetime.now().isoformat(),
            "last_login": datetime.now().isoformat()
        }])
        df = pd.concat([df, new_user], ignore_index=True)
        self.save_users(df)
        return True, "Signup successful. You've earned 10 points!"

    def login(self, username, password):
        df = self.load_users()
        user_row = df[df['username'] == username]
        if not user_row.empty:
            hashed = user_row.iloc[0]['password_hash']
            if self.verify_password(password, hashed):
                index = user_row.index[0]
                df.at[index, 'last_login'] = datetime.now().isoformat()
                df.at[index, 'points'] += 10  # Login reward
                self.save_users(df)
                return True, df.loc[index].to_dict()
        return False, "Invalid username or password."

    def add_points(self, username, points):
        df = self.load_users()
        if username in df['username'].values:
            idx = df[df['username'] == username].index[0]
            df.at[idx, 'points'] += points
            self.save_users(df)

    def get_user_points(self, username):
        df = self.load_users()
        if username in df['username'].values:
            return int(df[df['username'] == username]['points'].values[0])
        return 0

    def redeem_reward(self, username, cost):
        df = self.load_users()
        if username in df['username'].values:
            idx = df[df['username'] == username].index[0]
            if int(df.at[idx, 'points']) >= cost:
                df.at[idx, 'points'] -= cost
                self.save_users(df)
                return True
        return False


# === UI Render ===
def render_user_login_page():
    st.header("👤 User Portal")

    db = UserDatabase()

    # Clean up expired codes periodically
    db.cleanup_expired_codes()

    if 'user_logged_in' not in st.session_state:
        st.session_state.user_logged_in = False
        st.session_state.user_data = {}

    if not st.session_state.user_logged_in:
        tabs = st.tabs(["🔐 Login", "📝 Sign Up"])

        with tabs[0]:
            with st.form("user_login"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_btn = st.form_submit_button("Login")
                if login_btn:
                    success, user_or_msg = db.login(username, password)
                    if success:
                        st.session_state.user_logged_in = True
                        st.session_state.user_data = user_or_msg
                        st.success(f"Welcome back, {username}! (+10 points)")
                        st.rerun()
                    else:
                        st.error(user_or_msg)

        with tabs[1]:
            with st.form("user_signup"):
                new_user = st.text_input("Choose a Username")
                new_pass = st.text_input("Choose a Password", type="password")
                signup_btn = st.form_submit_button("Sign Up")
                if signup_btn:
                    success, msg = db.signup(new_user, new_pass)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
    else:
        user = st.session_state.user_data
        st.success(f"Logged in as: **{user['username']}**")
        points = db.get_user_points(user['username'])
        st.metric("🏆 Your Points", points)

        # Create tabs for different sections
        user_tabs = st.tabs(["🎁 Redeem Rewards", "🎫 Verify Code", "📋 My Codes", "🔧 Admin"])

        with user_tabs[0]:
            st.subheader("🎁 Redeem Your Points")
            reward_options = {
                "🎟️ Free Cinema Ticket (300 pts)": 300,
                "🛒 20% Off Supermarket Discount (200 pts)": 200,
                "☕ Free Coffee (100 pts)": 100,
            }
            reward = st.selectbox("Choose a reward:", list(reward_options.keys()))
            if st.button("Redeem"):
                cost = reward_options[reward]
                success = db.redeem_reward(user['username'], cost)
                if success:
                    st.success(f"✅ Successfully redeemed: {reward}")
                    st.rerun()
                else:
                    st.error("❌ Not enough points.")

        with user_tabs[1]:
            st.subheader("🎫 Verify Your Code")
            st.info("Enter the verification code you received from your parking reservation to earn bonus points!")

            with st.form("verify_code"):
                code_input = st.text_input("Enter Verification Code", placeholder="Enter 6-character code")
                verify_btn = st.form_submit_button("🔍 Verify Code")

                if verify_btn:
                    if code_input.strip():
                        success, message = db.verify_redeem_code(code_input.strip(), user['username'])
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.warning("Please enter a verification code.")

        with user_tabs[2]:
            st.subheader("📋 My Active Codes")
            active_codes = db.get_user_verification_codes(user['username'])

            if active_codes:
                st.success(f"You have {len(active_codes)} active verification code(s):")
                for code_info in active_codes:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.code(code_info['code'])
                    with col2:
                        st.write(f"**{code_info['points']} points**")
                    with col3:
                        st.write(f"⏰ {code_info['expires_in_minutes']} min left")
            else:
                st.info("No active verification codes. Complete a parking reservation to earn codes!")

        with user_tabs[3]:
            st.subheader("🔧 Admin - Generate Test Code")
            st.warning("⚠️ This is for testing purposes only")

            if st.button("🎯 Generate Test Verification Code"):
                test_code = db.generate_verification_code(user['username'], 50)
                st.success(f"Test code generated: **{test_code}**")
                st.info("This code is valid for 15 minutes and worth 50 points.")
                st.rerun()

        if st.button("🔓 Logout"):
            st.session_state.user_logged_in = False
            st.session_state.user_data = {}
            st.rerun()