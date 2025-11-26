"""
User Application Definition (app.py)

Define your application processes here.
The simulator (main.py) will load these tasks and combine them 
with the OS background processes selected in the GUI.
"""

def get_user_application_tasks():
    """
    Returns a list of dictionaries defining the User's Application Tasks.
    """

    # ==========================================
    # SCENARIO 1: Automotive Engine Control Unit (ECU)
    # Characteristics: High frequency, Hard Real-Time, Interrupt driven
    # ==========================================
    tasks_automotive = [
        # 1. Crankshaft Position Sensor (High Frequency)
        # Determines exactly where the pistons are.
        {
            "id": "ECU_Crank_Read",
            "wcet": 1,
            "period": 5,        # Very fast (simulates high RPM)
            "release": 0,
            "type": "periodic",
            "color": "blue"
        },

        # 2. Fuel Injection Calculation
        # Must happen immediately after reading position.
        {
            "id": "ECU_Fuel_Calc",
            "wcet": 2,
            "period": 5,
            "release": 0,
            "type": "periodic",
            "dependencies": ["ECU_Crank_Read"], 
            "deadline": 5,      # Hard Deadline: Miss this = Engine Misfire
            "color": "green"
        },

        # 3. Oxygen Sensor (Emissions Control)
        # Slower loop to adjust air/fuel mixture ratio.
        {
            "id": "ECU_O2_Adjust",
            "wcet": 3,
            "period": 20,       # Runs less often
            "release": 2,
            "type": "periodic",
            "color": "purple"
        },

        # 4. Dashboard CAN Bus Message
        # Sends RPM/Speed data to the dashboard. Low priority.
        {
            "id": "ECU_Dash_Update",
            "wcet": 2,
            "period": 40,
            "release": 5,
            "type": "periodic",
            "color": "gray"
        },

        # 5. Knock Sensor Interrupt (Pre-detonation)
        # Engine is "knocking" - must retard timing INSTANTLY to prevent damage.
        {
            "id": "ECU_Knock_Event",
            "wcet": 1,
            "release": 18,      # Random occurrence
            "deadline": 2,      # Ultra-critical response time
            "type": "interrupt",
            "color": "red"
        }
    ]

    # ==========================================
    # SCENARIO 2: Medical Infusion Pump
    # Characteristics: Safety Critical, Strict Dependencies, Reliability
    # ==========================================
    tasks_medical = [
        # 1. Flow Sensor Read
        # Measure how much fluid is actually moving.
        {
            "id": "Med_Flow_Sense",
            "wcet": 2,
            "period": 10,
            "release": 0,
            "type": "periodic",
            "color": "blue"
        },

        # 2. Dosage Calculation Algorithm
        # Compares flow against the prescribed rate.
        {
            "id": "Med_Dose_Calc",
            "wcet": 3,
            "period": 10,
            "release": 0,
            "type": "periodic",
            "dependencies": ["Med_Flow_Sense"], # Must wait for sensor
            "color": "green"
        },

        # 3. Motor Stepper Control
        # Physically moves the plunger based on calculation.
        {
            "id": "Med_Motor_Step",
            "wcet": 2,
            "period": 10,
            "release": 0,
            "type": "periodic",
            "dependencies": ["Med_Dose_Calc"], # Must wait for calc
            "deadline": 10,     # Must finish before next cycle
            "color": "orange"
        },

        # 4. System Integrity Check (Watchdog)
        # Ensures RAM/ROM is not corrupted.
        {
            "id": "Med_Integrity",
            "wcet": 4,
            "period": 50,       # Runs in background
            "release": 15,
            "type": "periodic",
            "color": "gray"
        },

        # 5. Air Bubble Detected! (Alarm)
        # Optical sensor sees air in the line. STOP PUMPING.
        {
            "id": "Med_Air_Alarm",
            "wcet": 1,
            "release": 35,
            "deadline": 3,      # Critical safety stop
            "type": "interrupt",
            "color": "red"
        }
    ]

    # ==========================================
    # SCENARIO 3: Mars Rover / Space Probe
    # Characteristics: Power/Thermal Management, Bursty Science Data
    # ==========================================
    tasks_space = [
        # 1. Thermal Management (Heaters)
        # Electronics will die if they get too cold. Highest Priority Periodic.
        {
            "id": "Rover_Thermal",
            "wcet": 2,
            "period": 15,
            "release": 0,
            "type": "periodic",
            "color": "red"
        },

        # 2. Navigation Camera (Hazard Avoidance)
        # Takes a picture to see rocks. Heavy processing.
        {
            "id": "Rover_Nav_Cam",
            "wcet": 6,          # Takes a long time to process image
            "period": 30,
            "release": 0,
            "type": "periodic",
            "dependencies": ["Rover_Thermal"], # Thermal runs first
            "color": "blue"
        },

        # 3. Path Planning (A* Algorithm)
        # Decides where to drive based on Nav Cam.
        {
            "id": "Rover_Path_Plan",
            "wcet": 5,
            "period": 30,
            "release": 0,
            "type": "periodic",
            "dependencies": ["Rover_Nav_Cam"],
            "color": "green"
        },

        # 4. Earth Uplink (Command Receive)
        # Listen for instructions from NASA.
        {
            "id": "Rover_Radio_Rx",
            "wcet": 2,
            "period": 60,       # Occurs rarely
            "release": 5,
            "type": "periodic",
            "color": "purple"
        },

        # 5. Dust Storm Warning (Sensor Interrupt)
        # Solar power dropping rapidly. Enter Safe Mode.
        {
            "id": "Rover_Safe_Mode",
            "wcet": 3,
            "release": 45,
            "deadline": 10,
            "type": "interrupt",
            "color": "orange"
        }
    ]

    # --- SELECT WHICH SCENARIO TO RUN ---
    # return tasks_automotive
    # return tasks_medical
    return tasks_space