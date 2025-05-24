from flask import Flask, render_template, request, jsonify
import pandas as pd
import google.generativeai as genai
import os
import json
from waitress import serve
from dotenv import load_dotenv
import logging # Ensure logging is imported

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            logger.critical("CRITICAL Error: GEMINI_API_KEY environment variable not set.") # Use logger
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        logger.critical(f"CRITICAL Error configuring Gemini: {e}", exc_info=True) # Use logger
        return None

# --- 3. Gemini Interaction (Revised for Chat Session & Contextual Follow-ups) ---
QUERY_PLAN_PROMPT_TEMPLATE = """
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

    prompt_for_this_turn = QUERY_PLAN_PROMPT_TEMPLATE.format(
        actual_columns_list_str=actual_columns_list_str,
        conceptual_to_actual_map_json_str=conceptual_to_actual_map_json_str,
        context_hint_text=context_hint_text,
        DEFAULT_DISPLAY_COLUMNS_CONCEPTUAL=DEFAULT_DISPLAY_COLUMNS_CONCEPTUAL,
        example_a_plan_str=example_a_plan_str,
        example_b_plan_str=example_b_plan_str,
        example_c_plan_str=example_c_plan_str,
        example_d_phd_progress_note_plan_str=example_d_phd_progress_note_plan_str,
        example_e_plan_contains_str=example_e_plan_contains_str,
        example_e_plan_exact_str=example_e_plan_exact_str,
        example_f_plan_str=example_f_plan_str,
        example_g_plan_str=example_g_plan_str,
        example_1_plan_str=example_1_plan_str,
        example_2_plan_revised_str=example_2_plan_revised_str,
        example_3_plan_str=example_3_plan_str,
        user_query=user_query
    )
    try:
        response = chat_session.send_message(prompt_for_this_turn)
        json_response_text = response.text.strip()
        if json_response_text.startswith("```json"):
            json_response_text = json_response_text[7:]
        if json_response_text.endswith("```"):
            json_response_text = json_response_text[:-3]
        plan = json.loads(json_response_text.strip())
        return plan
    except json.JSONDecodeError as e:
        logger.error(f"AI response was not valid JSON: {json_response_text}. Error: {e}", exc_info=True)
        return {"is_answerable": False, "reason_if_not_answerable": "AI response was not valid JSON."}
    except Exception as e:
        logger.error(f"Gemini API error or other error in generate_query_plan: {e}", exc_info=True)
        return {"is_answerable": False, "reason_if_not_answerable": f"Gemini API error. Please check server logs."}

def clean_filter_value(value):
    if not isinstance(value, str):
        return value
    value = value.lower().strip()
    # Order might matter here if suffixes can overlap
    for suffix in [" form", " note", " script"]:
        if value.endswith(suffix):
            value = value[:-len(suffix)]
            break # Removes only the first found suffix
    return value

# --- 4. DataFrame Query Logic (Executor) ---
def execute_query_plan(df, column_map, plan):
    global LAST_SUCCESSFUL_PLAN_CONTEXT, LAST_PRIMARY_ENTITY_CONTEXT

    if df is None or column_map is None:
        logger.error("DataFrame or column map not loaded in execute_query_plan.")
        return {"type": "text", "content": "DataFrame or column map not loaded."}
    if not plan.get("is_answerable"):
        return {
            "type": "text",
            "content": f"I cannot answer that. Reason: {plan.get('reason_if_not_answerable', 'AI determined it is not answerable.')}"
        }

    operation = plan.get("operation")
    filters_plan = plan.get("filters", [])
    
    filtered_df = df.copy()
    current_primary_entity_from_filters = None # Context derived from filters in THIS query

    if filters_plan:
        # Consider the first filter's target as a potential primary entity from this query
        if filters_plan[0].get("value") and filters_plan[0].get("column_conceptual_name"):
            current_primary_entity_from_filters = {
                'type': filters_plan[0].get("column_conceptual_name"),
                'value': filters_plan[0].get("value")
            }
            logger.info(f"Tentative primary entity from current query's filters: {current_primary_entity_from_filters}")

        for f_spec in filters_plan:
            concept_col = f_spec.get("column_conceptual_name")
            actual_col_name = column_map.get(concept_col)
            match_type = f_spec.get("match_type", "contains").lower()
            if match_type == "equals": # Normalize "equals" to "exact"
                match_type = "exact"
            if match_type not in ("exact", "contains", "not_exact"):
                logger.warning(f"Unsupported match_type: '{match_type}'.")
                return {"type": "text", "content": f"Unsupported match_type: '{match_type}'."}
            
            value_to_filter_original = f_spec.get("value") 
            value_to_filter = clean_filter_value(value_to_filter_original)
            logger.info(f"Filtering: conceptual_col='{concept_col}', actual_col='{actual_col_name}', match='{match_type}', original_val='{value_to_filter_original}', cleaned_val='{value_to_filter}'")

            if not actual_col_name:
                logger.error(f"Filter Error: Conceptual column '{concept_col}' not mapped.")
                return {"type": "text", "content": f"Filter Error: Conceptual column '{concept_col}' not mapped."}
            if value_to_filter is None: 
                logger.error(f"Filter Error: No value provided for filtering '{concept_col}'.")
                return {"type": "text", "content": f"Filter Error: No value for filtering '{concept_col}'."}
            if actual_col_name not in filtered_df.columns:
                logger.error(f"Filter Error: Mapped column '{actual_col_name}' not in DataFrame.")
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
                logger.error(f"Error during filtering on '{actual_col_name}': {e}", exc_info=True)
                return {"type": "text", "content": f"Error during filtering on '{actual_col_name}': {e}"}
    
    if filtered_df.empty:
        logger.info("Filtering resulted in an empty DataFrame.")
        LAST_SUCCESSFUL_PLAN_CONTEXT = None 
        if current_primary_entity_from_filters: 
             LAST_PRIMARY_ENTITY_CONTEXT = current_primary_entity_from_filters
             logger.info(f"Updated LAST_PRIMARY_ENTITY_CONTEXT to {LAST_PRIMARY_ENTITY_CONTEXT} from current query's filters despite empty results.")
        else: # If no filters in current query led to empty, clear context
            LAST_PRIMARY_ENTITY_CONTEXT = None # Clear context if no results and no specific filter context
            logger.info("Cleared LAST_PRIMARY_ENTITY_CONTEXT due to empty results without specific filter context.")
        return {"type": "text", "content": "No data found matching your specified filter criteria."}

    # If we have results, this plan was successful.
    LAST_SUCCESSFUL_PLAN_CONTEXT = plan
    logger.info(f"Set LAST_SUCCESSFUL_PLAN_CONTEXT.")

    # Determine LAST_PRIMARY_ENTITY_CONTEXT for the next turn.
    # Precedence:
    # 1. A single, specific entity resulting from a "filter_and_list" operation (most specific).
    # 2. A specific entity mentioned in the filters of the current query.
    # 3. The general class of items targeted by an operation if no more specific entity is identified.

    new_primary_context = None # Temporary variable for clarity

    # 1. Check for single specific item from "filter_and_list"
    if operation == "filter_and_list":
        # Prioritize script if it's the only one in results
        script_col_actual = column_map.get("script_name_conceptual")
        if script_col_actual and script_col_actual in filtered_df.columns:
            unique_scripts_in_result = filtered_df[script_col_actual].dropna().unique()
            if len(unique_scripts_in_result) == 1:
                new_primary_context = {'type': 'script_name_conceptual', 'value': str(unique_scripts_in_result[0])}
                logger.info(f"Context from single script result: {new_primary_context}")
        
        # Could add similar logic for other key entities like 'form_name_conceptual' if desired
        # For now, single script is a strong indicator.

    # 2. If no single specific item, check for context from current query's filters
    if new_primary_context is None and current_primary_entity_from_filters:
        new_primary_context = current_primary_entity_from_filters
        logger.info(f"Context from current query's filters: {new_primary_context}")

    # 3. If still no context, derive from operation type (for count or list_unique without filters)
    if new_primary_context is None:
        if operation == "count_items" and plan.get("count_target_conceptual"):
            new_primary_context = {'type': plan.get("count_target_conceptual"), 'value': 'items previously counted'}
            logger.info(f"Context from count_items operation: {new_primary_context}")
        elif operation == "list_unique_values" and not filters_plan and plan.get("list_unique_target_conceptual"):
            # 'not filters_plan' is important: if there were filters, current_primary_entity_from_filters would have been set
            new_primary_context = {'type': plan.get("list_unique_target_conceptual"), 'value': 'all unique values listed'}
            logger.info(f"Context from list_unique_values operation (no filters): {new_primary_context}")
    
    if new_primary_context:
        LAST_PRIMARY_ENTITY_CONTEXT = new_primary_context
        logger.info(f"Updated LAST_PRIMARY_ENTITY_CONTEXT to {LAST_PRIMARY_ENTITY_CONTEXT}")
    # If no new context was derived, LAST_PRIMARY_ENTITY_CONTEXT retains its previous value (or None if never set)
    # This might be okay, or we might want to explicitly clear it if no context is derivable from current successful query.
    # For now, this logic prioritizes new, relevant context.

    # --- Actual data processing based on operation ---
    if operation == "filter_and_list":
        display_concepts = plan.get("display_columns_conceptual", DEFAULT_DISPLAY_COLUMNS_CONCEPTUAL)
        display_actual_cols = [column_map.get(c) for c in display_concepts if column_map.get(c) and column_map.get(c) in filtered_df.columns]

        if not display_actual_cols and not filtered_df.empty: 
            logger.warning(f"Specified display columns {display_concepts} resulted in no valid columns. Falling back.")
            display_actual_cols = [col for col in filtered_df.columns if col in column_map.values()] 
            if not display_actual_cols: 
                display_actual_cols = filtered_df.columns.tolist()
        
        logger.info(f"Displaying columns: {display_actual_cols}")
        if not filtered_df.empty and display_actual_cols:
            df_to_display = filtered_df[display_actual_cols].drop_duplicates()
            try:
                html_table = df_to_display.to_html(index=False, classes=['results-table'], border="0", justify='left')
                return {"type": "html", "content": f"Results:\n{html_table}"}
            except Exception as e:
                logger.error(f"Error converting DataFrame to HTML: {e}", exc_info=True) 
                return {"type": "text", "content": "Error displaying results as table. Data found, but format error."}
        elif filtered_df.empty: 
            return {"type": "text", "content": "No data found."} # Should have been caught earlier
        else: 
            logger.warning("Data found, but no valid columns to display.")
            return {"type": "text", "content": "Data found, but no valid columns to display."}

    elif operation == "count_items":
        count_target_concept = plan.get("count_target_conceptual")
        count_distinct = plan.get("count_distinct", False) 
        
        if not filtered_df.empty: 
            if not count_target_concept: 
                count = len(filtered_df)
                return {"type": "text", "content": f"Found {count} records matching your criteria."}

            actual_count_col = column_map.get(count_target_concept)
            if not actual_count_col or actual_count_col not in filtered_df.columns:
                logger.error(f"Count Error: Target column '{count_target_concept}' (actual: {actual_count_col}) not mapped/found in DataFrame.")
                return {"type": "text", "content": f"Count Error: Target column '{count_target_concept}' not mapped/found."}

            if count_distinct:
                count = filtered_df[actual_count_col].nunique()
                entity_name = count_target_concept.replace("_conceptual", "").replace("_name", "").capitalize() + "s"
                if count == 1:
                    entity_name = entity_name[:-1] 
                return {"type": "text", "content": f"Found {count} distinct {entity_name} ({actual_count_col}) matching your criteria."}
            else:
                count = filtered_df[actual_count_col].notna().sum() 
                entity_name = count_target_concept.replace("_conceptual", "").replace("_name", "").capitalize()
                return {"type": "text", "content": f"Found {count} items/records where '{entity_name}' ({actual_count_col}) is present, matching criteria."}
        else: 
            return {"type": "text", "content": "No data to count matching your criteria."}


    elif operation == "list_unique_values":
        list_target_concept = plan.get("list_unique_target_conceptual")
        actual_list_col = column_map.get(list_target_concept)
        if not actual_list_col or actual_list_col not in df.columns: 
            logger.error(f"List Unique Error: Target column '{list_target_concept}' (actual: {actual_list_col}) not mapped or not in original DataFrame.")
            return {"type": "text", "content": f"List Unique Error: Target column '{list_target_concept}' not mapped."}

        df_to_use_for_unique = filtered_df if filters_plan and not filtered_df.empty else df
        
        if not df_to_use_for_unique.empty:
            if actual_list_col not in df_to_use_for_unique.columns: 
                 logger.error(f"List Unique Error: Column '{actual_list_col}' not in data source for unique values.")
                 return {"type": "text", "content": f"List Unique Error: Column '{actual_list_col}' not in data for unique values."}
            unique_values = df_to_use_for_unique[actual_list_col].dropna().unique()
            return {"type": "text", "content": f"Unique values for '{actual_list_col}':\n" + "\n".join(sorted(map(str, unique_values)))}
        else:
            return {"type": "text", "content": f"No data to list unique values for '{actual_list_col}'."}

    else:
        logger.warning(f"Unsupported operation: '{operation}'.")
        return {"type": "text", "content": f"Unsupported operation: '{operation}'."}


# --- Flask App Setup ---
app = Flask(__name__)

# --- Data Loading and Initial Setup ---
EHR_DATA_FILE = os.environ.get("EHR_DATA_FILE", "Scriptlink.xlsx") 
try:
    EHR_DF = pd.read_excel(EHR_DATA_FILE)
    logger.info(f"Successfully loaded data from {EHR_DATA_FILE}")
except FileNotFoundError:
    logger.critical(f"CRITICAL ERROR: Data file '{EHR_DATA_FILE}' not found. The application will not work correctly.")
    EHR_DF = None 
except Exception as e:
    logger.critical(f"CRITICAL ERROR: Failed to load data from '{EHR_DATA_FILE}': {e}", exc_info=True)
    EHR_DF = None


COLUMN_MAP = {}
if EHR_DF is not None:
    for concept_key, aliases in COLUMN_ALIASES.items():
        found_mapping = False
        for alias in aliases:
            for col in EHR_DF.columns:
                if col.replace(" ", "").lower() == alias.replace(" ", "").lower():
                    COLUMN_MAP[concept_key] = col
                    logger.info(f"Mapped conceptual column '{concept_key}' to actual column '{col}' using alias '{alias}'.")
                    found_mapping = True
                    break 
            if found_mapping:
                break 
        if not found_mapping:
             logger.warning(f"Conceptual column '{concept_key}' could not be mapped to any column in the Excel file using aliases: {aliases}")
else:
    logger.warning("EHR_DF is None, skipping COLUMN_MAP creation.")


GEMINI_MODEL = configure_gemini()
CHAT_SESSION = GEMINI_MODEL.start_chat(history=[]) if GEMINI_MODEL else None
if not CHAT_SESSION:
    logger.warning("Chat session could not be initialized (Gemini model might be None).")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def handle_send_message():
    global LAST_SUCCESSFUL_PLAN_CONTEXT, LAST_PRIMARY_ENTITY_CONTEXT 

    if not CHAT_SESSION or EHR_DF is None or not COLUMN_MAP:
        logger.error("Chatbot not fully initialized (Chat session, EHR_DF, or COLUMN_MAP missing).")
        return jsonify({'reply_type': 'text', 'reply': 'Chatbot not fully initialized. Please wait or check server logs.'}), 500

    user_query = request.json.get('message')
    if not user_query:
        logger.warning("Received empty message from user.")
        return jsonify({'reply_type': 'text', 'reply': 'No message provided'}), 400
    
    logger.info(f"Received user query: '{user_query}'")

    small_talk_response = handle_small_talk(user_query)
    if small_talk_response:
        logger.info("Handled as small talk.")
        return jsonify(small_talk_response)

    actual_columns_list_str = ", ".join(EHR_DF.columns)
    response_data = {"reply_type": "text", "reply": "An error occurred processing your request."} 

    try:
        simple_show_previous_cmds = ["what was that again?", "show that again", "repeat that"]
        if user_query.lower().strip() in simple_show_previous_cmds and LAST_SUCCESSFUL_PLAN_CONTEXT:
            logger.info("Executing last successful plan due to repeat command.")
            result = execute_query_plan(EHR_DF, COLUMN_MAP, LAST_SUCCESSFUL_PLAN_CONTEXT)
        else:
            logger.info("Generating new query plan.")
            plan = generate_query_plan_with_chat(
                CHAT_SESSION, user_query, actual_columns_list_str, COLUMN_MAP, LAST_PRIMARY_ENTITY_CONTEXT
            )
            logger.info(f"Generated plan: {plan}")
            result = execute_query_plan(EHR_DF, COLUMN_MAP, plan)
        
        logger.info(f"Execution result: {result}")
        response_data['reply_type'] = result.get('type', 'text')
        response_data['reply'] = result.get('content', 'No response from executor.')

    except Exception as e:
        logger.error(f"Unhandled server error in handle_send_message: {e}", exc_info=True) 
        response_data['reply'] = f"Server error: {str(e)}" 
        response_data['reply_type'] = 'text'

    return jsonify(response_data)

if __name__ == '__main__':
    PORT = int(os.environ.get("PORT", 8080))
    if GEMINI_MODEL and EHR_DF is not None and CHAT_SESSION: 
        logger.info(f"Starting Flask server with Waitress on http://0.0.0.0:{PORT}") 
        serve(app, host='0.0.0.0', port=PORT)
    else:
        logger.critical("Application cannot start due to missing critical components (Gemini, EHR Data, or Chat Session). Check logs.")