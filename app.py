"""
User Application Definition (app.py)

Define your application processes here.
The simulator (main.py) will load these tasks and combine them 
with the OS background processes selected in the GUI.
"""

def get_user_application_tasks():
    """
    Returns a list of dictionaries defining the User's Application Tasks.
    
    Fields:
    - id: Name of the task
    - wcet: Worst Case Execution Time
    - period: (Optional) Period for periodic tasks
    - deadline: (Optional) Relative deadline
    - release: Arrival time
    - type: "periodic", "aperiodic", or "interrupt"
    - dependencies: List of task IDs that must finish before this one starts
    """
    
    tasks = [
        # --- Example: A Video Streaming App ---
        
        # 1. Network Data Fetcher (Periodic, High Priority typically)
        {
            "id": "App_NetFetch",
            "wcet": 2,
            "period": 10,
            "release": 0,
            "type": "periodic",
            "color": "blue"
        },
        
        # 2. Video Decoder (Depends on Fetch, Periodic)
        {
            "id": "App_Decoder",
            "wcet": 3,
            "period": 10,
            "release": 0,
            "type": "periodic",
            "dependencies": ["App_NetFetch"], # Decoder waits for Fetch
            "color": "orange"
        },
        
        # 3. Audio Processing (Periodic, needs to be fast)
        {
            "id": "App_AudioProc",
            "wcet": 1,
            "period": 5, 
            "release": 0,
            "type": "periodic",
            "color": "green"
        },
        
        # 4. UI Update / Render (Main Thread)
        {
            "id": "App_UI_Render",
            "wcet": 2,
            "period": 15,
            "release": 2,
            "type": "periodic",
            "color": "purple"
        },
        
        # 5. User Interaction (Aperiodic - e.g., Pause button pressed)
        {
            "id": "App_Input_Event",
            "wcet": 1,
            "release": 25,
            "deadline": 5, # Needs quick response
            "type": "interrupt",
            "color": "red"
        }
    ]

    tasks2 = [
        # --- The Core Game Loop (Sequential Dependencies) ---

        # 1. Input Manager (Must read keyboard/mouse first)
        {
            "id": "Game_Input",
            "wcet": 1,          # Very fast check
            "period": 16,       # Target: ~60 FPS (16ms)
            "release": 0,
            "type": "periodic",
            "color": "blue"
        },

        # 2. Physics Engine (Collision & Movement)
        # MUST happen after Input, so the player moves where they aimed.
        {
            "id": "Game_Physics",
            "wcet": 3,          # Heavy math calculations
            "period": 16,
            "release": 0,
            "type": "periodic",
            "dependencies": ["Game_Input"],
            "color": "green"
        },

        # 3. Renderer (Draw the frame)
        # Cannot draw until Physics is done calculating positions.
        {
            "id": "Game_Render",
            "wcet": 4,          # Heavy GPU/CPU overhead
            "period": 16,
            "release": 0,
            "type": "periodic",
            "dependencies": ["Game_Physics"], 
            "deadline": 16,     # Soft Deadline: miss this = lag spike/frame drop
            "color": "purple"
        },

        # --- Independent / Background Systems ---

        # 4. Enemy AI Logic (Pathfinding)
        # Runs slower than the graphics (e.g., 10 times a second) to save CPU.
        {
            "id": "Game_Enemy_AI",
            "wcet": 5,          # Complex A* pathfinding algorithms
            "period": 40,       # Slower tick rate
            "release": 5,
            "type": "periodic",
            "color": "red"
        },

        # 5. Network Sync (Multiplayer)
        # Sends player coordinates to the server.
        {
            "id": "Game_Net_Sync",
            "wcet": 2,
            "period": 30,
            "release": 10,
            "type": "periodic",
            "color": "cyan"
        },

        # 6. Explosion Event (Aperiodic)
        # A grenade went off! Needs immediate processing for particle effects.
        {
            "id": "Game_Explosion",
            "wcet": 2,
            "release": 45,      # Occurs later in the simulation
            "deadline": 5,      # Visuals must appear instantly
            "type": "interrupt",
            "color": "orange"
        }
    ]  
    tasks3 = [
        # --- Multimedia (Soft Real-Time) ---
        # These apps need consistent CPU time to avoid audio stuttering.

        # 1. Spotify/Music Stream (Audio Buffer Fill)
        # Requires frequent, short bursts of CPU to fill the audio buffer.
        {
            "id": "Laptop_Music_Stream",
            "wcet": 1,          # Very quick
            "period": 5,        # Runs very often (high frequency)
            "release": 0,
            "type": "periodic",
            "color": "green"
        },

        # 2. Zoom/Teams Meeting (Video Encoding)
        # Needs a good chunk of CPU regularly to process video frames.
        {
            "id": "Laptop_Zoom_Call",
            "wcet": 3,
            "period": 10,       # 10 ticks ~ 1 video frame interval
            "release": 0,
            "type": "periodic",
            "deadline": 10,     # If this is missed, video freezes/lags
            "color": "blue"
        },

        # --- The Work Workflow (Dependent Tasks) ---
        
        # 3. IDE Auto-Save (Text Editor)
        # Saves the file before the compiler can run.
        {
            "id": "Laptop_IDE_Save",
            "wcet": 1,
            "period": 40,       # Autosave every 40 ticks
            "release": 2,
            "type": "periodic",
            "color": "gray"
        },

        # 4. Code Compilation (gcc/make)
        # Heavy CPU usage. Cannot start until the file is saved.
        {
            "id": "Laptop_Compiler",
            "wcet": 8,          # Compiling takes a long time!
            "period": 40,
            "release": 2,
            "type": "periodic",
            "dependencies": ["Laptop_IDE_Save"], # Waits for save
            "color": "orange"
        },

        # --- Interruptions ---

        # 5. Slack/Discord Notification
        # A message arrives and demands immediate UI focus (pop-up).
        {
            "id": "Laptop_Slack_Ping",
            "wcet": 1,
            "release": 23,      # Random time during the work
            "deadline": 3,      # Needs to show up immediately
            "type": "interrupt",
            "color": "red"
        },
        
        # 6. Windows Update / Virus Scan (Background)
        # Low priority annoyance that runs in the background.
        {
            "id": "Laptop_Sys_Update",
            "wcet": 5,
            "period": 60,       # Runs rarely
            "release": 15,
            "type": "periodic",
            "color": "purple"
        }
    ]
    
    
    
    return tasks