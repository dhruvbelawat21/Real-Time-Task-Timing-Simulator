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
    
    return tasks