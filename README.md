# EHR Scriptlink Chatbot

## Overview
The EHR Scriptlink Chatbot is a web-based application that allows users to query information about Electronic Health Record (EHR) web service scripts using natural language. It leverages a generative AI model to understand user questions and retrieve relevant data from an Excel file.

## Features
*   **Natural Language Queries**: Ask questions about EHR script data in plain English.
*   **Contextual Understanding**: Remembers the context of your conversation for relevant follow-up questions.
*   **Small Talk**: Responds to simple greetings and pleasantries.
*   **Rich Display**: Shows query results, including data tables, directly in the chat interface.
*   **Dark Mode**: Adapts to your system's preferred color scheme for comfortable viewing.

## How it Works
*   **Frontend**: A simple web interface built with HTML, CSS, and JavaScript provides the chat UI.
*   **Backend**: A Flask (Python) application handles user requests, processes queries, and interacts with the AI model.
*   **Data Source**: Information is read from a `Scriptlink.xlsx` file, which should contain details about EHR scripts, forms, services, namespaces, etc.
*   **Query Processing**:
    1.  User input is sent from the frontend to the Flask backend.
    2.  Google's Gemini AI model processes the natural language query.
    3.  The AI generates a structured JSON "query plan".
    4.  This plan is executed by the backend using `pandas` to filter and retrieve data from the `Scriptlink.xlsx` DataFrame.
    5.  The results are sent back to the frontend for display.

## Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create and Activate a Virtual Environment**:
    (Recommended to avoid conflicts with global Python packages)
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    Ensure you have the `requirements.txt` file from the repository.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**:
    Create a file named `.env` in the root directory of the project. Add your Gemini API key to this file:
    ```
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```
    Replace `"YOUR_API_KEY_HERE"` with your actual API key.

5.  **Prepare the Data File**:
    *   Place your Excel data file named `Scriptlink.xlsx` in the root directory of the project.
    *   This file should contain the EHR script information you want to query. The application expects columns that can be conceptually mapped to terms like "Form Name", "Script Name", "Field Name", "Service Name", and "Namespace" (see `COLUMN_ALIASES` in `app.py`).

## Running the Application

1.  **Start the Flask Server**:
    Once dependencies are installed and the `.env` file is configured, run:
    ```bash
    python app.py
    ```

2.  **Access the Chatbot**:
    Open your web browser and navigate to:
    `http://127.0.0.1:8080`

## Usage
Once the application is running and you've opened it in your browser:
*   You'll see a chat interface.
*   Type your questions about EHR scripts, forms, services, etc., into the input box at the bottom.
*   Press Enter or click "Send".
*   The chatbot's response will appear in the chat window.

**Example Queries**:
*   "What scripts are on the 'Patient Demographics' form?"
*   "How many services are related to 'Billing Claims'?"
*   "List all unique forms containing the word 'progress'."
*   "Show me scripts in the 'cws' namespace."

## Column Mapping
The application uses a predefined mapping located in `app.py` (under `COLUMN_ALIASES`) to understand user queries. This mapping links conceptual terms (like 'Form Name', 'Script Name') to the actual column names in your `Scriptlink.xlsx` file. If your Excel file has different column names, you may need to adjust this mapping in `app.py` for the chatbot to correctly interpret your data.

## `.gitignore`
The repository includes a `.gitignore` file that excludes common Python artifacts (like `__pycache__/`) and sensitive files (like `.env`) from version control.
