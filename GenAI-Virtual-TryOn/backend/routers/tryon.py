from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from utils.base64_helpers import array_buffer_to_base64
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import traceback
import base64

load_dotenv()

router = APIRouter()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

@router.post("/try-on")
async def try_on(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    instructions: str = Form(""),
    model_type: str = Form(""),
    gender: str = Form(""),
    garment_type: str = Form(""),
    style: str = Form(""),
):
    try:
        
        MAX_IMAGE_SIZE_MB = 10
        ALLOWED_MIME_TYPES = {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/heic",
            "image/heif",
        }

        if person_image.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type for person_image: {person_image.content_type}"
            )

        user_bytes = await person_image.read()

        size_in_mb_for_person_image = len(user_bytes) / (1024 * 1024)
        if size_in_mb_for_person_image > MAX_IMAGE_SIZE_MB:
            raise HTTPException(status_code=400, detail="Image exceeds 10MB size limit for person_image")
        
        if cloth_image.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type for cloth_image: {cloth_image.content_type}"
            )

        cloth_bytes = await cloth_image.read()

        size_in_mb_for_cloth_image = len(cloth_bytes) / (1024 * 1024)
        if size_in_mb_for_cloth_image > MAX_IMAGE_SIZE_MB:
            raise HTTPException(status_code=400, detail="Image exceeds 10MB size limit for cloth_image")


        user_b64 = array_buffer_to_base64(user_bytes)
        cloth_b64 = array_buffer_to_base64(cloth_bytes)

        prompt = f"""
            {{
            "objective": "Generate a photorealistic virtual try-on image, seamlessly integrating a specified clothing item onto a person while rigidly preserving their facial identity, the clothing's exact appearance, and placing them in a completely new, distinct background.",
            "task": "High-Fidelity Virtual Try-On with Identity/Garment Preservation and Background Replacement", 

            "inputs": {{
            "person_image": {{
                "description": "Source image containing the target person. Used *primarily* for identity (face, skin tone), pose, body shape, hair, and accessories. The original background will be DISCARDED.",
                "id": "input_1"
            }},
            "garment_image": {{
                "description": "Source image containing the target clothing item. May include a model, mannequin, or be flat-lay. Used *strictly* for the clothing's visual properties (color, style, texture, pattern).",
                "id": "input_2"
            }}
            }},

            "processing_steps": [
            "Isolate the clothing item from 'garment_image' (input_2), discarding any original model, mannequin, or background. Extract exact color, pattern, texture, and style information.",
            "Identify the person (face, body shape, skin tone), pose, hair, and accessories from 'person_image' (input_1).",
            "Segment the person from the original background in 'person_image'.",
            "Determine the desired new background based on 'background_preference' or generate a suitable default.",
            "Analyze lighting cues from 'person_image' to inform initial lighting on the subject, but adapt lighting for consistency with the *new* background."
            ],

            "output_requirements": {{
            "description": "Generate a single, high-resolution image where the person from 'person_image' appears to be naturally and realistically wearing the clothing item from 'garment_image', situated within a **completely new and different background**.",
            "format": "Image (e.g., PNG, JPG)",
            "quality": "Photorealistic, free of obvious artifacts, blending issues, or inconsistencies between subject, garment, and the new background."
            }},

            "core_constraints": {{
            "identity_lock": {{
                "priority": "ABSOLUTE CRITICAL",
                "instruction": "Maintain the **PERFECT** facial identity, features, skin tone, and expression of the person from 'person_image'. **ZERO alterations** to the face are permitted. Treat the head region (including hair) as immutable unless directly and logically occluded by the garment. DO NOT GUESS OR HALLUCINATE FACIAL FEATURES."
            }},
            "garment_fidelity": {{
                "priority": "ABSOLUTE CRITICAL",
                "instruction": "Preserve the **EXACT** color (hue, saturation, brightness), pattern, texture, material properties, and design details of the clothing item from 'garment_image'. **ZERO deviations** in style, color, or visual appearance are allowed. Render the garment precisely as depicted in input_2."
            }},
            "background_replacement": {{
                "priority": "CRITICAL",
                "instruction": "Generate a **COMPLETELY NEW and DIFFERENT** background that is distinct from the original background in 'person_image'. The new background should be photorealistic and contextually plausible for a person/fashion image unless otherwise specified by 'background_preference'. Ensure the person is seamlessly integrated into this new environment. **NO elements** from the original background should remain visible."
            }},
            "pose_preservation": {{
                "priority": "HIGH",
                "instruction": "Retain the **exact** body pose and positioning of the person from 'person_image'."
            }},
            "realistic_integration": {{
                "priority": "HIGH",
                "instruction": "Simulate physically plausible draping, folding, and fit of the garment onto the person's body according to their shape and pose. Ensure natural interaction with the body within the context of the *new* background."
            }},
            "lighting_consistency": {{
                "priority": "HIGH",
                "instruction": "Apply lighting, shadows, and highlights to the rendered garment AND the person that are **perfectly consistent** with the direction, intensity, and color temperature implied by the **NEW background**. Adjust subject lighting subtly if necessary to match the new scene, but prioritize maintaining a natural look consistent with the original subject's lighting where possible."
            }}
            }},

            "additional_constraints": {{
            "body_proportion_accuracy": "Scale the garment accurately to match the person's body proportions.",
            "occlusion_handling": "Render natural and correct occlusion where the garment covers parts of the body, hair, or existing accessories from 'person_image'. Preserve visible unique features (tattoos, scars) unless occluded.",
            "hair_and_accessory_integrity": "Maintain hair and non-clothing accessories (glasses, jewelry, hats) from 'person_image' unless logically occluded by the new garment. Integrate them seamlessly with the person and the new background.",
            "texture_and_detail_rendering": "Render fine details (e.g., embroidery, seams, buttons, lace, sheer fabric properties) from the garment with high fidelity.",
            "scene_coherence": "Ensure the person logically fits within the generated background environment (e.g., appropriate scale, perspective, interaction with ground plane if applicable)."
            }},

            "edge_case_handling": {{
            "tight_fitting_clothing": "Accurately depict fabric stretch and conformity to body contours.",
            "transparent_sheer_clothing": "Realistically render transparency, showing underlying skin tone or layers appropriately.",
            "complex_garment_geometry": "Handle unusual shapes, layers, or asymmetrical designs with correct draping.",
            "unusual_poses": "Ensure garment drape remains physically plausible even in non-standard or dynamic poses.",
            "garment_partially_out_of_frame": "Render the visible parts of the garment correctly; do not hallucinate missing sections.",
            "low_resolution_inputs": "Maximize detail preservation but prioritize realistic integration over inventing details not present in the inputs.",
            "mismatched_lighting_inputs": "Prioritize generating a coherent lighting environment based on the **NEW background**, adapting the garment and slightly adjusting the person's apparent lighting for a unified final image. Avoid harsh lighting clashes."
            }},

            "prohibitions": [
            "DO NOT alter the person's facial features, identity, expression, or skin tone.",
            "DO NOT modify the intrinsic color, pattern, texture, or style of the clothing item.",
            "DO NOT retain ANY part of the original background from 'person_image'.",
            "DO NOT change the person's pose.",
            "DO NOT introduce elements not present in the input images (person, garment) except for the generated background and necessary shadows/lighting adjustments for integration.",
            "DO NOT hallucinate or guess facial details; if obscured, maintain the integrity of visible parts based on identity lock.",
            "DO NOT generate a background that is stylistically jarring or contextually nonsensical without explicit instruction via 'background_preference'."
            ]
            }}

            You are a virtual fashion stylist.
            Create a realistic try-on visualization of the uploaded clothing onto the person image.
            Match the following context:
            - Model Type: {model_type}
            - Gender: {gender}
            - Garment Type: {garment_type}
            - Style: {style}
            - Special Instructions: {instructions}

           Return image of try on and a short caption or summary of how the outfit looks and fits. Also include suggestions for improvement.
        
        """
               
        print(model_type)
        print(gender)
        print(garment_type)
        print(style)
        print(instructions)
        
        print(prompt)

        contents=[
            prompt,
            types.Part.from_bytes(
                data=user_b64,
                mime_type= person_image.content_type,
            ),
            types.Part.from_bytes(
                data=cloth_b64,
                mime_type= cloth_image.content_type,
            ),
        ]        
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
            )
        )


        print(response)
        
        image_data = None
        text_response = "No Description available."
        if response.candidates and len(response.candidates) > 0:
            parts = response.candidates[0].content.parts

            if parts:
                print("Number of parts in response:", len(parts))

                for part in parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        image_data = part.inline_data.data
                        image_mime_type = getattr(part.inline_data, "mime_type", "image/png")
                        print("Image data received, length:", len(image_data))
                        print("MIME type:", image_mime_type)

                    elif hasattr(part, "text") and part.text:
                        text_response = part.text
                        preview = (text_response[:100] + "...") if len(text_response) > 100 else text_response
                        print("Text response received:", preview)
            else:
                print("No parts found in the response candidate.")
        else:
            print("No candidates found in the API response.")

        image_url = None
        if image_data:
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            image_url = f"data:{image_mime_type};base64,{image_base64}"
        else:
            image_url = None
    
        return JSONResponse(
        content={
            "image": image_url,
            "text": text_response,
        }
        )

    except Exception as e:
        print(f"Error in /api/try-on endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
