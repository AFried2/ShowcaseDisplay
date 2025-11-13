import streamlit as st

# --- Page Configuration ---
# This must be the first Streamlit command in your script
st.set_page_config(
    page_title="App Dashboard",
    page_icon="👋",
    layout="wide",
)

# --- Project Data ---
# A dictionary where keys are project names and values are details.
# You'll need to create preview images (e.g., 'Project1/preview.png')
# and place them in the correct folders for this to work.
PROJECTS = {
    "Project 1": {
        "description": "A brief description of what Project 1 does. This app showcases data analysis.",
        "image": "Text-Predicting-RNN/preview.png", # <-- UPDATED this path
        "page_name": "Project_1" # This must match the filename in the /pages folder (without the number prefix)
    },
    # --- I've commented out the placeholder projects. ---
    # --- You can add them back as you build them. ---
    # "Project 2": {
    #     "description": "Project 2 is a machine learning model explorer. Interact with different models.",
    #     "image": "Project2/preview.png",
    #     "page_name": "Project_2"
    # },
    # "Project 3": {
    #     "description": "A cool app that generates images using AI. Fun to play with.",
    #     "image": "httpsReadMe.png", # Example of a different path
    #     "page_name": "Project_3"
    # },
    # "Project 4": {
    #     "description": "This project visualizes complex financial data in real-time.",
    #     "image": "https://placehold.co/600x400/000000/FFFFFF?text=Project+4",
    #     "page_name": "Project_4"
    # },
    # "Project 5": {
    #     "description": "Description for project 5.",
    #     "image": "https://placehold.co/600x400/333333/FFFFFF?text=Project+5",
    #     "page_name": "Project_5"
    # },
    # "Project 6": {
    #     "description": "Description for project 6.",
    #     "image": "https://placehold.co/600x400/555555/FFFFFF?text=Project+6",
    #     "page_name": "Project_6"
    # },
}

# --- Dashboard UI ---
st.title("Streamlit Project Showcase")
st.write("Welcome to my collection of Streamlit apps! Use the sidebar to navigate to any app, or check them out below.")
st.divider()

# --- Grid Layout ---
# Create a 3-column layout
cols = st.columns(3)
col_index = 0

for project_name, details in PROJECTS.items():
    # Place each project card in the next available column
    with cols[col_index]:
        st.subheader(project_name)
        
        # Display the image with a fallback
        try:
            # --- FIX: Changed use_column_width to use_container_width ---
            st.image(details["image"], use_container_width=True, caption=details["description"])
        except Exception as e:
            st.error(f"Could not load image: {details['image']}\n{e}", icon="🖼️")
            # --- FIX: Added https:// to the fallback URL and fixed width ---
            st.image("https://placehold.co/600x400/FF0000/FFFFFF?text=Image+Not+Found", use_container_width=True)

        st.write(details["description"])
        
        # This button is optional, as navigation is in the sidebar.
        # It's here for show. To make it functional, you'd need st.switch_page
        # (which requires Streamlit 1.33+).
        # if st.button(f"Go to {project_name}", key=f"btn_{project_name}"):
        #     try:
        #         st.switch_page(f"pages/{details['page_name']}.py")
        #     except Exception as e:
        #         st.error(f"Could not switch to page. Make sure 'pages/{details['page_name']}.py' exists.")

        st.divider()

    # Move to the next column, wrap around back to 0
    col_index = (col_index + 1) % 3

st.caption("All projects built with Streamlit.")


