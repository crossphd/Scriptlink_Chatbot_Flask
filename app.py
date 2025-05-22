from flask import Flask, render_template, request, jsonify
import pandas as pd
import google.generativeai as genai
import os
import json
from waitress import serve
from dotenv import load_dotenv

load_dotenv()

def handle_small_talk(user_query):
    text = user_query.strip().lower()
    ehr_keywords = ['script', 'form', 'service', 'field', 'count', 'list', 'diagnosis', 'progress note', 'phd'] 
    # Only respond to small talk if no EHR keywords in query
    if any(keyword in text for keyword in ehr_keywords):
        return None  # Likely a data query, not small talk

    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    thanks = ["thank you", "thanks", "thx", "appreciate it"]
    how_are_you = ["how are you", "how's it going", "how are things"]

    if any(greet in text for greet in greetings):
        return {"reply_type": "text", "reply": "Hello! How can I assist you with EHR scripts today?"}
    if any(thank in text for thank in thanks):
        return {"reply_type": "text", "reply": "You're welcome! Happy to help."}
    if any(phrase in text for phrase in how_are_you):
        return {"reply_type": "text", "reply": "I'm doing well, thanks for asking! What would you like to know about the scripts?"}
    
    return None

# --- Configuration for Column Name Mapping ---
COLUMN_ALIASES = {
    "form_name_conceptual": ["FormName", "Form Name", "Form", "Forms", "EHR Form"],
    "script_name_conceptual": ["ScriptName", "Script Name", "Script", "Scripts", "EHR Script"],
    "field_name_conceptual": ["FieldName", "Field Name", "Field", "Fields", "Form Field", "EHR Field", "Field_Name", "Field_ID"],
    "service_name_conceptual": ["ServiceName", "Service Name", "Service", "Services", "EHR Service"],
    "namespace_conceptual": ["Namespace", "Namespaces", "Env", "Environment"]
}
DEFAULT_DISPLAY_COLUMNS_CONCEPTUAL = ["script_name_conceptual", "form_name_conceptual", "field_name_conceptual", "service_name_conceptual", "namespace_conceptual"]

# Store context from the last interaction (these will be global for this simple single-user app)
LAST_SUCCESSFUL_PLAN_CONTEXT = None
LAST_PRIMARY_ENTITY_CONTEXT = None

# --- 1. Configuration & Setup ---
def configure_gemini():
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("CRITICAL Error: GEMINI_API_KEY environment variable not set.")
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        print(f"CRITICAL Error configuring Gemini: {e}")
        return None

# --- 3. Gemini Interaction (Revised for Chat Session & Contextual Follow-ups) ---
def generate_query_plan_with_chat(chat_session, user_query, actual_columns_list_str, conceptual_to_actual_map_dict, last_primary_entity_context):
    conceptual_to_actual_map_json_str = json.dumps(conceptual_to_actual_map_dict) 
    
    context_hint_text = ""
    if last_primary_entity_context:
        entity_type_conceptual = last_primary_entity_context['type']
        entity_value = last_primary_entity_context['value']
        context_hint_text = (
            f"NOTE: In prior conversation, the main referenced entity is a '{entity_type_conceptual}' with value '{entity_value}'. "
            "When the user uses pronouns like 'it', 'that script', or 'that form', "
            "assume they refer to this entity unless otherwise specified. "
            "Use this context to disambiguate ambiguous queries."
        )

    # JSON examples as strings
    example_a_plan_str = json.dumps({
        "is_answerable": True,
        "operation": "filter_and_list",
        "filters": [
            {"column_conceptual_name": "script_name_conceptual", "match_type": "exact", "value": "ScriptA"},
        ],
        "display_columns_conceptual": ["form_name_conceptual", "service_name_conceptual"]
    })
    example_b_plan_str = json.dumps({
        "is_answerable": True,
        "operation": "count_items",
        "filters": [{"column_conceptual_name": "form_name_conceptual", "match_type": "exact", "value": "Form Z"}],
        "count_target_conceptual": "service_name_conceptual",
        "count_distinct": True
    })
    example_c_plan_str = json.dumps({
        "is_answerable": True,
        "operation": "count_items",
        "filters": [{"column_conceptual_name": "form_name_conceptual", "match_type": "contains", "value": "special use"}],
        "count_target_conceptual": "script_name_conceptual"
    })
    example_d_phd_progress_note_plan_str = json.dumps({
        "is_answerable": True,
        "operation": "filter_and_list",
        "filters": [{"column_conceptual_name": "form_name_conceptual", "match_type": "contains", "value": "phd"}],
        "display_columns_conceptual": ["script_name_conceptual", "service_name_conceptual"],
        "reasoning": "User query 'phd progress note' refers to a form likely containing 'phd'. Using 'contains' with core entity 'phd'."
    })
    example_e_plan_contains_str = json.dumps({
        "is_answerable": True,
        "operation": "filter_and_list",
        "filters": [{"column_conceptual_name": "form_name_conceptual", "match_type": "contains", "value": "diagnosis"}],
        "display_columns_conceptual": ["form_name_conceptual"]
    })
    example_e_plan_exact_str = json.dumps({
        "is_answerable": True,
        "operation": "filter_and_list",
        "filters": [{"column_conceptual_name": "form_name_conceptual", "match_type": "exact", "value": "Diagnosis"}],
        "display_columns_conceptual": ["form_name_conceptual"]
    })
    example_f_plan_str = json.dumps({
        "is_answerable": True,
        "operation": "count_items",
        "filters": [{"column_conceptual_name": "namespace_conceptual", "match_type": "contains", "value": "cws"}],
        "count_target_conceptual": "form_name_conceptual",
        "count_distinct": True
    })
    example_g_plan_str = json.dumps({
        "is_answerable": True,
        "operation": "count_items",
        "filters": [{"column_conceptual_name": "form_name_conceptual", "match_type": "contains", "value": "progress note"}],
        "count_target_conceptual": "script_name_conceptual",
        "count_distinct": True
    })
    example_1_plan_str = json.dumps({
        "is_answerable": True,
        "operation": "filter_and_list",
        "filters": [{"column_conceptual_name": "form_name_conceptual", "match_type": "exact", "value": "Patient Demographics"}],
        "display_columns_conceptual": ["script_name_conceptual", "field_name_conceptual", "service_name_conceptual"]
    })
    example_2_plan_revised_str = json.dumps({
        "is_answerable": True,
        "operation": "count_items",
        "filters": [{"column_conceptual_name": "form_name_conceptual", "match_type": "exact", "value": "Billing Claims"}],
        "count_target_conceptual": "script_name_conceptual",
        "count_distinct": True
    })
    example_3_plan_str = json.dumps({
        "is_answerable": True,
        "operation": "filter_and_list",
        "filters": [{"column_conceptual_name": "form_name_conceptual", "match_type": "contains", "value": "progress note"}],
        "display_columns_conceptual": ["form_name_conceptual"]
    })

    prompt_for_this_turn = f"""
You are an AI assistant that helps users query a DataFrame about EHR web service scripts.
Actual columns in DataFrame: {actual_columns_list_str}.
Conceptual to actual column mapping: {conceptual_to_actual_map_json_str}.
When specifying a column in your plan, ALWAYS use the 'conceptual_name' from the mapping.

Your primary goal: Understand the user's LATEST question.
--- CRITICAL CONTEXT HANDLING ---
1. Examine the full CONVERSATION HISTORY (which you have access to via the chat mechanism) to understand the current context.
2. Resolve pronouns (it, that, those, these script, that form etc.) or anaphoric references based on entities mentioned in PREVIOUS, RELEVANT turns of the conversation.
3. If the user asks a follow-up question like "is that script on any other forms?" or "what about service X for it?", identify the specific entity (e.g., script name 'ScriptABC', form name 'FormXYZ') from the history.
4. Use this identified entity to formulate the NEW JSON plan for the CURRENT user question.
{context_hint_text}

 --- FILTER VALUE EXTRACTION ---
IMPORTANT FOR FILTER VALUES:
- When the user query includes descriptive terms like "form", "note", "script" AFTER a more specific identifier (e.g., "diagnosis form", "phd progress note", "billing script"), your main task is to isolate the core identifying part (e.g., "diagnosis", "phd", "phd note", "billing").
- This core identifier SHOULD BE USED as the "value" in your filter.
- For instance, if user says "phd progress note":
    - Core identifiers could be "phd" or "phd note".
    - The "value" in your filter should be "phd" or "phd note", NOT the full "phd progress note".
- ALWAYS set "match_type" to "contains" when using such an extracted core entity, unless you are absolutely certain the core entity is an exact, complete name found in the data.
- If the user says just "phd note", extract "phd" or "phd note" as the value.
- If user says "special use progress note", extract "special use" or "special use note" as the value.

--- COUNTING INSTRUCTIONS ---
IMPORTANT FOR COUNTING:
- When the user asks "how many forms...", "how many scripts...", "number of unique services...", your "operation" should be "count_items".
- The "count_target_conceptual" should be the conceptual name of the item they want to count (e.g., "form_name_conceptual" for forms, "script_name_conceptual" for scripts).
- Set "count_distinct" to true if the user implies they want a count of unique items (e.g., "how many unique forms", "how many different scripts"). If they just say "how many forms" or "how many scripts", it's usually safer to assume they mean distinct entities.

--- JSON PLAN STRUCTURE ---
Generate a JSON plan:
- "is_answerable": true/false. If false, provide "reason_if_not_answerable".
- "operation": "filter_and_list", "count_items", "list_unique_values".
- "filters": List of {{ "column_conceptual_name": "...", "match_type": "exact" or "contains" or "not_exact", "value": "..." }}. Curly braces in this line must be doubled for f-string.
- "display_columns_conceptual": For "filter_and_list". Default: {DEFAULT_DISPLAY_COLUMNS_CONCEPTUAL}.
- "count_target_conceptual": For "count_items".
- "count_distinct": (boolean, optional for "count_items").
- "list_unique_target_conceptual": For "list_unique_values".
- "reasoning": (Optional) Your brief reasoning for the plan, especially how you resolved context.

--- EXAMPLES ---
Examples of resolving context and partial names:
A. Previous Interaction: User: "What scripts are on 'Form X'?" -> Bot shows 'ScriptA'.
   LATEST User Question: "Is that script on any other forms?"
   Your reasoning for the LATEST question: User means 'ScriptA' from history. Plan should find forms for 'ScriptA', potentially excluding 'Form X'.
   JSON Plan (for LATEST question): {example_a_plan_str}

B. Previous Interaction: User: "How many services on 'Form Y'?" -> Bot gives a count.
   LATEST User Question: "What about 'Form Z'?"
   Your reasoning: User wants the same type of query (count services) but for 'Form Z'.
   JSON Plan: {example_b_plan_str}


C. User: "How many scripts are on the special use note?" (DataFrame has "Special Use Progress Note")
   Your reasoning: "special use note" -> core entity "special use". Match type "contains".
   JSON Plan: {example_c_plan_str}

D. User: "what scripts are on the phd progress note" (DataFrame has "PhD PsyD SW MFT Progress Note")
   Your reasoning: User query "phd progress note" implies the core entity is "phd" or "phd note". Use "contains".
   JSON Plan: {example_d_phd_progress_note_plan_str}

E. User: "Is there a diagnosis form?" (DataFrame has a form named "Diagnosis")
   Your reasoning: "diagnosis form" -> core entity "diagnosis".
   JSON Plan (using contains for robustness): {example_e_plan_contains_str}
   JSON Plan (using exact if confident): {example_e_plan_exact_str}

F. User: "how many forms have cws namespace"
   Your reasoning: User wants to count distinct forms filtered by namespace.
   JSON Plan: {example_f_plan_str}

G. User: "how many scripts are on progress note forms"
   Your reasoning: User wants to count distinct scripts found on any form containing 'progress note'.
   JSON Plan: {example_g_plan_str}

Original examples of single-turn plans:
1. User: "What scripts are on the 'Patient Demographics' form?"
   JSON: {example_1_plan_str}
2. User: "How many distinct scripts are on 'Billing Claims'?"
   JSON: {example_2_plan_revised_str}
3. User: "Show forms with 'progress note'."
   JSON: {example_3_plan_str}

LATEST User Question: "{user_query}"
JSON Plan:
"""
    try:
        response = chat_session.send_message(prompt_for_this_turn)
        json_response_text = response.text.strip()
        if json_response_text.startswith("```json"): json_response_text = json_response_text[7:]
        if json_response_text.endswith("```"): json_response_text = json_response_text[:-3]
        plan = json.loads(json_response_text.strip())
        return plan
    except json.JSONDecodeError as e:
        return {"is_answerable": False, "reason_if_not_answerable": "AI response was not valid JSON."}
    except Exception as e:
        last_response_text = "N/A"
        if chat_session.history and chat_session.history[-1].role == "model":
            if chat_session.history[-1].parts: last_response_text = chat_session.history[-1].parts[0].text
        return {"is_answerable": False, "reason_if_not_answerable": f"Gemini API error. Please check server logs."}

def clean_filter_value(value):
    if not isinstance(value, str):
        return value
    value = value.lower().strip()
    for suffix in [" form", " note", " script"]:
        if value.endswith(suffix):
            value = value[:-len(suffix)]
            break
    return value

# --- 4. DataFrame Query Logic (Executor) ---
def execute_query_plan(df, column_map, plan):
    global LAST_SUCCESSFUL_PLAN_CONTEXT, LAST_PRIMARY_ENTITY_CONTEXT

    if df is None or column_map is None:
        return {"type": "text", "content": "DataFrame or column map not loaded."}
    if not plan.get("is_answerable"):
        return {
            "type": "text",
            "content": f"I cannot answer that. Reason: {plan.get('reason_if_not_answerable', 'AI determined it is not answerable.')}"
        }

    operation = plan.get("operation")
    filters_plan = plan.get("filters", [])
    
    filtered_df = df.copy()
    current_primary_entity_from_filters = None

    if filters_plan:
        if filters_plan[0].get("value") and filters_plan[0].get("column_conceptual_name"):
            current_primary_entity_from_filters = {
                'type': filters_plan[0].get("column_conceptual_name"),
                'value': filters_plan[0].get("value")
            }
        for f_spec in filters_plan:
            concept_col = f_spec.get("column_conceptual_name")
            actual_col_name = column_map.get(concept_col)
            match_type = f_spec.get("match_type", "contains").lower()
            if match_type == "equals":
                match_type = "exact"
            if match_type not in ("exact", "contains", "not_exact"):
                return {"type": "text", "content": f"Unsupported match_type: '{match_type}'."}
            value_to_filter = clean_filter_value(f_spec.get("value"))

            if not actual_col_name:
                return {"type": "text", "content": f"Filter Error: Conceptual column '{concept_col}' not mapped."}
            if value_to_filter is None:
                return {"type": "text", "content": f"Filter Error: No value for filtering '{concept_col}'."}
            if actual_col_name not in filtered_df.columns:
                return {"type": "text", "content": f"Filter Error: Mapped column '{actual_col_name}' not in DataFrame."}

            try:
                col_as_str = filtered_df[actual_col_name].astype(str)
                if match_type == "exact":
                    filtered_df = filtered_df[col_as_str.str.lower() == str(value_to_filter).lower()]
                elif match_type == "contains":
                    filtered_df = filtered_df[col_as_str.str.contains(str(value_to_filter), case=False, na=False)]
                elif match_type == "not_exact":
                    filtered_df = filtered_df[col_as_str.str.lower() != str(value_to_filter).lower()]
            except Exception as e:
                return {"type": "text", "content": f"Error during filtering on '{actual_col_name}': {e}"}
        
        if filtered_df.empty:
            LAST_SUCCESSFUL_PLAN_CONTEXT = None
            LAST_PRIMARY_ENTITY_CONTEXT = current_primary_entity_from_filters
            return {"type": "text", "content": "No data found matching your specified filter criteria."}

    LAST_SUCCESSFUL_PLAN_CONTEXT = plan
    if current_primary_entity_from_filters:
        LAST_PRIMARY_ENTITY_CONTEXT = current_primary_entity_from_filters

    if not filtered_df.empty:
        if operation == "filter_and_list":
            script_col_actual = column_map.get("script_name_conceptual")
            if script_col_actual and script_col_actual in filtered_df.columns:
                unique_scripts_in_result = filtered_df[script_col_actual].dropna().unique()
                if len(unique_scripts_in_result) == 1:
                    LAST_PRIMARY_ENTITY_CONTEXT = {
                        'type': 'script_name_conceptual',
                        'value': str(unique_scripts_in_result[0])
                    }
    elif operation == "list_unique_values" and not filters_plan:
        if plan.get("list_unique_target_conceptual"):
            LAST_PRIMARY_ENTITY_CONTEXT = {'type': plan.get("list_unique_target_conceptual"), 'value': 'all unique values listed'}

    if operation == "filter_and_list":
        display_concepts = plan.get("display_columns_conceptual", DEFAULT_DISPLAY_COLUMNS_CONCEPTUAL)
        display_actual_cols = [column_map.get(c) for c in display_concepts if column_map.get(c) and column_map.get(c) in filtered_df.columns]

        if not display_actual_cols and not filtered_df.empty:
            display_actual_cols = [col for col in filtered_df.columns if col in column_map.values()]
            if not display_actual_cols:
                display_actual_cols = filtered_df.columns.tolist()

        if not filtered_df.empty and display_actual_cols:
            df_to_display = filtered_df[display_actual_cols].drop_duplicates()
            try:
                html_table = df_to_display.to_html(index=False, classes=['results-table'], border="0", justify='left')
                return {"type": "html", "content": f"Results:\n{html_table}"}
            except Exception as e:
                print(f"Error converting DataFrame to HTML: {e}")
                return {"type": "text", "content": "Error displaying results as table. Data found, but format error."}

        elif filtered_df.empty:
            return {"type": "text", "content": "No data found."}
        else:
            return {"type": "text", "content": "Data found, but no valid columns to display."}

    elif operation == "count_items":
        count_target_concept = plan.get("count_target_conceptual")
        count_distinct = plan.get("count_distinct", False)
        if not filtered_df.empty:
            if not count_target_concept:
                return {"type": "text", "content": f"Found {len(filtered_df)} records matching your criteria."}
            actual_count_col = column_map.get(count_target_concept)
            if not actual_count_col or actual_count_col not in filtered_df.columns:
                return {"type": "text", "content": f"Count Error: Target column '{count_target_concept}' not mapped/found."}
            if count_distinct:
                count = filtered_df[actual_count_col].nunique()
                entity_name = count_target_concept.replace("_conceptual", "").replace("_name", "").capitalize() + "s"
                if count == 1:
                    entity_name = entity_name[:-1]
                return {"type": "text", "content": f"Found {count} distinct {entity_name} ({actual_count_col}) matching your criteria."}
            else:
                count = len(filtered_df)
                entity_name = count_target_concept.replace("_conceptual", "").replace("_name", "").capitalize()
                return {"type": "text", "content": f"Found {count} items/records where '{entity_name}' is present, matching criteria."}
        else:
            return {"type": "text", "content": "No data to count matching your criteria."}

    elif operation == "list_unique_values":
        list_target_concept = plan.get("list_unique_target_conceptual")
        actual_list_col = column_map.get(list_target_concept)
        if not actual_list_col or actual_list_col not in df.columns:
            return {"type": "text", "content": f"List Unique Error: Target column '{list_target_concept}' not mapped."}

        df_to_use_for_unique = filtered_df if filters_plan and not filtered_df.empty else df

        if not df_to_use_for_unique.empty:
            if actual_list_col not in df_to_use_for_unique.columns:
                return {"type": "text", "content": f"List Unique Error: Column '{actual_list_col}' not in data for unique values."}
            unique_values = df_to_use_for_unique[actual_list_col].dropna().unique()
            return {"type": "text", "content": f"Unique values for '{actual_list_col}':\n" + "\n".join(sorted(map(str, unique_values)))}
        else:
            return {"type": "text", "content": f"No data to list unique values for '{actual_list_col}'."}

    else:
        return {"type": "text", "content": f"Unsupported operation: '{operation}'."}


# --- Flask App Setup ---
app = Flask(__name__)

EHR_DF = pd.read_excel("Scriptlink.xlsx")
COLUMN_MAP = {}
for concept_key, aliases in COLUMN_ALIASES.items():
    for alias in aliases:
        for col in EHR_DF.columns:
            if col.replace(" ", "").lower() == alias.replace(" ", "").lower():
                COLUMN_MAP[concept_key] = col
                break
        if concept_key in COLUMN_MAP:
            break

GEMINI_MODEL = configure_gemini()
CHAT_SESSION = GEMINI_MODEL.start_chat(history=[]) if GEMINI_MODEL else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def handle_send_message():
    global LAST_SUCCESSFUL_PLAN_CONTEXT, LAST_PRIMARY_ENTITY_CONTEXT

    if not CHAT_SESSION or EHR_DF is None or not COLUMN_MAP:
        return jsonify({'reply_type': 'text', 'reply': 'Chatbot not fully initialized. Please wait or check server logs.'}), 500

    user_query = request.json.get('message')
    if not user_query:
        return jsonify({'reply_type': 'text', 'reply': 'No message provided'}), 400

    # Check small talk first
    small_talk_response = handle_small_talk(user_query)
    if small_talk_response:
        return jsonify(small_talk_response)

    actual_columns_list_str = ", ".join(EHR_DF.columns)
    response_data = {"reply_type": "text", "reply": "An error occurred processing your request."}

    try:
        simple_show_previous_cmds = ["what was that again?", "show that again", "repeat that"]
        if user_query.lower().strip() in simple_show_previous_cmds and LAST_SUCCESSFUL_PLAN_CONTEXT:
            result = execute_query_plan(EHR_DF, COLUMN_MAP, LAST_SUCCESSFUL_PLAN_CONTEXT)
        else:
            plan = generate_query_plan_with_chat(
                CHAT_SESSION, user_query, actual_columns_list_str, COLUMN_MAP, LAST_PRIMARY_ENTITY_CONTEXT
            )
            result = execute_query_plan(EHR_DF, COLUMN_MAP, plan)

        response_data['reply_type'] = result.get('type', 'text')
        response_data['reply'] = result.get('content', 'No response from executor.')

    except Exception as e:
        import traceback
        traceback.print_exc()
        response_data['reply'] = f"Server error: {str(e)}"
        response_data['reply_type'] = 'text'

    return jsonify(response_data)

if __name__ == '__main__':
    if GEMINI_MODEL:
        print("Starting Flask server on http://127.0.0.1:8080")
        serve(app, host='0.0.0.0', port=8080)
    else:
        print("Gemini failed to initialize.")
