import os
import instructor
from groq import Groq
from pipeline.models import EventList
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMExtractor:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("Warning: GROQ_API_KEY environment variable not set. LLM extraction will fail.")
        
        # Patch the Groq client with Instructor to easily enforce Pydantic structured outputs
        self.client = instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.JSON)
        # Using Llama 4 Scout because it has high reasoning + high daily token limits (500K TPD)
        self.model_name = "meta-llama/llama-4-scout-17b-16e-instruct"

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60), 
        stop=stop_after_attempt(5),
        reraise=True
    )
    def extract_events(self, text: str, known_assets: list[dict], zero_shot_category: str) -> dict | None:
        """Extracts structured events bridging unstructured text and the Pydantic schema using Groq."""
        
        # Prepare known assets context to help LLaMA normalize vessel names
        vessel_names = [asset['attributes']['assetName'] for asset in known_assets if asset['attributes'].get('assetType') == 'vessel']
        vessels_context = ", ".join(vessel_names) if vessel_names else "None explicitly registered in metadata."

        prompt = f"""
        You are an expert maritime intelligence analyst.
        Analyze the following maritime news article and extract all distinct maritime events.
        
        CRITICAL INSTRUCTIONS:
        1. Resolve all pronouns and generic references (e.g., 'the carrier', 'the vessel') to their specific named entities before extracting.
        2. Normalize vessel names using this provided reference list of known vessels in the article (if applicable): [{vessels_context}]. Only use vessel names from this list if you are referring to the same ship.
        3. Determine the 'incident_type'. An initial automated pass suggested it might be: "{zero_shot_category}". Validate or correct this.
        4. If a detail is not explicitly mentioned in the text, return 'null' for that field. Do not infer or invent details.
        
        SEMANTIC ENRICHMENT RULES (Requirement 4):
        - `executive_summary`: Write a highly professional, 150-word intelligence brief summarizing the entirety of the article context.
        - `risk_level`: CRITICAL (loss of life, massive spill, war zone), HIGH (major port blockage, ship loss), MEDIUM (delays, minor damage), LOW (routine, resolved).
        - `impact_scope`: GLOBAL (affects global supply chains/prices), REGIONAL (affects a specific sea/coastline), LOCAL (affects only the immediate port/vessel).
        - `strategic_relevance_tags`: Generate 2-4 short tags like 'Supply Chain Shock', 'Sanctions Evasion', 'Piracy Trend', or 'Environmental Fines'.
        - Boolean Flags: Set to true if the article contains state actors (geopolitics), navies/military (defense), or OFAC/sanctions/smuggling (sanction sensitive).

        ARTICLE TEXT:
        {text}
        """

        try:
            # instructor intercepts the call, forcing LLaMA to return JSON that perfectly matches EventList
            structured_response = self.client.chat.completions.create(
                model=self.model_name,
                response_model=EventList,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, # Keep it deterministic
            )
            
            # Convert the Pydantic model back to a dictionary so the rest of the pipeline remains unchanged
            return structured_response.model_dump()
            
        except Exception as e:
            if "429" in str(e):
                print(f"⚠️ Groq Rate Limit Hit. Waiting and retrying...")
                raise e # Throw it back up for tenacity to catch and sleep
            else:
                print(f"❌ Error during LLM extraction: {e}")
                return None
