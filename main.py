from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from huggingface_hub import login
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import logging
import re
from typing import Optional

import os
from dotenv import load_dotenv

load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Interview Questions Generator API",
    description="API para generar preguntas de entrevista basadas en presentaciones de candidatos",
    version="1.0.0"
)

# Modelos Pydantic para request/response
class InterviewRequest(BaseModel):
    presentation: str
    topic: Optional[str] = ""
    age: Optional[str] = ""

class InterviewResponse(BaseModel):
    questions: str
    status: str = "success"

# Variables globales para el modelo
llm_chain = None
HF_TOKEN = os.getenv("HF_TOKEN")

def extract_json_from_response(text: str) -> str:
    """
    Extrae el array JSON de la respuesta del modelo - versión simplificada
    """
    try:
        # Buscar el primer [ y el último ]
        start = text.find('[')
        end = text.rfind(']')
        
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
            
            # Intentar parsear como JSON
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list) and len(parsed) > 0:
                    # Verificar que no sean preguntas genéricas
                    valid_questions = []
                    for q in parsed:
                        if isinstance(q, str) and len(q) > 20 and not is_generic_question(q):
                            valid_questions.append(q)
                    
                    if valid_questions:
                        # Devolver JSON limpio, sin escapes innecesarios
                        return json.dumps(valid_questions[:3], ensure_ascii=False, indent=None)
            except json.JSONDecodeError:
                pass
        
        # Si falla, usar preguntas por defecto
        default_questions = [
            "Can you describe the most challenging aspect of developing the BCP virtual assistant with NestJS and Angular?",
            "How do you approach integrating AI technologies like WatsonX Assistant with traditional web frameworks?",
            "Given your experience with multiple cloud platforms, how do you decide which one to use for a specific project?"
        ]
        
        return json.dumps(default_questions, ensure_ascii=False, indent=None)
        
    except Exception as e:
        logger.error(f"Error extrayendo JSON: {str(e)}")
        # Último recurso
        fallback_questions = [
            "Tell me about a complex fullstack project you've developed",
            "How do you integrate AI and automation in your development workflow?",
            "Describe your experience working with cloud platforms and microservices"
        ]
        return json.dumps(fallback_questions, ensure_ascii=False, indent=None)

def is_generic_question(question: str) -> bool:
    """Detecta si una pregunta es demasiado genérica"""
    if not isinstance(question, str):
        return True
        
    generic_indicators = [
        "question 1", "question 2", "question 3",
        "pregunta 1", "pregunta 2", "pregunta 3",
        "insert", "example", "sample", "question about"
    ]
    return any(indicator in question.lower() for indicator in generic_indicators)

# Función para inicializar el modelo
def initialize_model():
    """Inicializa el modelo y la cadena LLM"""
    global llm_chain
    
    try:
        logger.info("Iniciando sesión en Hugging Face...")
        login(token=HF_TOKEN)
        
        logger.info("Cargando modelo Phi-4-mini-instruct...")
        # Configurar parámetros del pipeline para mejor control
        model = pipeline(
            "text-generation",
            model="gpt2",
            max_new_tokens=300,  # Aumentar para preguntas más completas
            truncation=True,
            do_sample=True,
            temperature=0.8,     # Más creatividad
            top_p=0.9,
            pad_token_id=50256
        )
        
        logger.info("Configurando LangChain pipeline...")
        llm = HuggingFacePipeline(pipeline=model)
        
        template = """<|user|>
You are an expert programming interviewer. Based on the candidate's background, create exactly 3 technical interview questions in English. Return only a simple JSON array of strings.

Candidate: {presentation}

Return format - only this structure with no extra formatting:
["question about their specific experience", "question about their technologies", "question about their projects"]<|end|>
<|assistant|>
"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["presentation"]
        )
        
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        logger.info("Modelo inicializado correctamente")
        
    except Exception as e:
        logger.error(f"Error al inicializar el modelo: {str(e)}")
        raise e

# Evento de startup para cargar el modelo
@app.on_event("startup")
async def startup_event():
    """Carga el modelo al iniciar la aplicación"""
    logger.info("Iniciando aplicación...")
    initialize_model()

# Endpoint de salud
@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado de la API"""
    return {
        "status": "healthy",
        "message": "API is running",
        "model_loaded": llm_chain is not None
    }

# Endpoint principal para generar preguntas
@app.post("/generate-questions", response_model=InterviewResponse)
async def generate_interview_questions(request: InterviewRequest):
    """
    Genera preguntas de entrevista basadas en la presentación del candidato
    """
    try:
        if llm_chain is None:
            raise HTTPException(
                status_code=503, 
                detail="Modelo no inicializado. Inténtalo más tarde."
            )
        
        if not request.presentation.strip():
            raise HTTPException(
                status_code=400,
                detail="La presentación no puede estar vacía"
            )
        
        logger.info("Generando preguntas de entrevista...")
        
        # Generar respuesta usando el modelo
        response = llm_chain.invoke({"presentation": request.presentation})
        
        # Extraer y limpiar el texto de la respuesta
        generated_text = response.get("text", "")
        
        # Intentar extraer solo la parte JSON de la respuesta
        cleaned_response = extract_json_from_response(generated_text)
        
        logger.info("Preguntas generadas exitosamente")
        
        return InterviewResponse(
            questions=cleaned_response,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al generar preguntas: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )

# Endpoint con presentación por defecto (manteniendo la funcionalidad original)
@app.post("/generate-default-questions", response_model=InterviewResponse)
async def generate_default_questions():
    """
    Genera preguntas usando la presentación por defecto de Leandro Ramos
    """
    default_presentation = """
    Mi nombre es Leandro Ramos, soy fullstack web con más de 5 años de experiencia, he trabajado en clientes como ecommerce, retail, startups. Mi enfoque es en la automatizacion e Inteligencia Artificial. En cuanto a las tecnologías que manejo, tengo experiencia con: nodejs, express, nestjs, react, angular, nextjs, postgresql, mongodb, aws, azure, ibm cloud, watsonx assistant, langchain, vertex. Respecto a mi experiencia laboral: He trabajado como desarrollador fullstack cognitivo apoyando con el desarrollo del asistente virtual de BCP, el cual es una empresa de banca de gran impacto, fue desarrollada con tecnologias como: nestjs, angular, watsonx assistant, modelos como mistral-large
    """
    
    request = InterviewRequest(presentation=default_presentation)
    return await generate_interview_questions(request)

# Endpoint para obtener información sobre la API
@app.get("/")
async def root():
    """Información básica de la API"""
    return {
        "message": "Interview Questions Generator API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Verificar estado de la API",
            "/generate-questions": "Generar preguntas personalizadas",
            "/generate-default-questions": "Generar preguntas con presentación por defecto",
            "/docs": "Documentación interactiva"
        }
    }

# Manejo de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error no manejado: {str(exc)}")
    return {
        "status": "error",
        "message": "Error interno del servidor",
        "detail": str(exc)
    }

# Para ejecutar en desarrollo: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# O si prefieres ejecutar con python main.py:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )