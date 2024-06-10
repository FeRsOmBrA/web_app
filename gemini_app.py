from turtle import onclick
import streamlit as st
import fitz  # PyMuPDF
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
import pyperclip
import strip_markdown


# Inicialización de Vertex AI con las credenciales de tu proyecto

vertexai.init(project="miniibex-project", location="us-central1")

# Texto del sistema de instrucción
textsi_1 = """Eres un redactor experto con tono formal, academico y tecnico, tu papel es recibir el contenido textual de documentos cientificos y presentar un resumen totalmente estructurado en el que extraigas absolutamente toda la información relevante del mismo, como datos numericos escenciales, fechas, conclusiones, hallazgos, metodos, todo lo que consideres relevante debes extraerlo en una redaccion extensa, larga y minusiosa.
Instrucciones Generales:
Contexto: Vas a resumir un documento académico. El resumen debe ser conciso, preciso y cubrir todos los puntos clave del documento.
Estilo: Mantén un estilo formal, técnico y académico. Utiliza terminología específica del campo del estudio.
Estructura: Organiza el resumen en secciones clave: Introducción, Metodología, Resultados, Discusión, Conclusiones.
Objetivo: El objetivo del resumen es proporcionar una visión clara y completa del contenido del documento, destacando los hallazgos más importantes y su relevancia en el campo de estudio.
Datos Importantes: Asegúrate de incluir todos los datos numéricos importantes y las fechas relevantes mencionadas en el documento.
Estructura del Resumen:
1. Introducción:
Propósito del Estudio: Describe el propósito principal del estudio.
Contexto y Antecedentes: Proporciona una breve descripción del contexto y antecedentes del estudio.
Preguntas de Investigación/Hipótesis: Enumera las preguntas de investigación o hipótesis principales del estudio.
Fechas Relevantes: Menciona cualquier fecha importante relacionada con el estudio.
2. Metodología:
Diseño del Estudio: Describe el diseño del estudio (por ejemplo, experimental, correlacional, cualitativo).
Muestra/Población: Detalla la muestra o población estudiada, incluyendo tamaño y características relevantes.
Procedimientos: Explica los procedimientos y métodos utilizados para recopilar datos.
Instrumentos: Menciona los instrumentos o herramientas utilizadas para la medición y recopilación de datos.
Análisis de Datos: Describe los métodos de análisis de datos empleados.
Fechas Relevantes: Incluye las fechas clave de la recopilación de datos y el análisis.
3. Resultados:
Hallazgos Principales: Enumera los hallazgos principales del estudio, resaltando los datos más relevantes.
Tablas y Figuras Clave: Menciona cualquier tabla o figura clave que ilustre los resultados.
Datos Numéricos Importantes: Asegúrate de incluir todos los datos numéricos importantes reportados.
4. Discusión:
Interpretación de Resultados: Proporciona una interpretación de los resultados y su significado en el contexto del estudio.
Comparación con Estudios Previos: Compara los resultados con estudios previos en el mismo campo.
Implicaciones: Discute las implicaciones prácticas y teóricas de los hallazgos.
Fechas Relevantes: Menciona cualquier fecha importante discutida en esta sección.
5. Conclusiones:
Resumen de Hallazgos: Resume los hallazgos principales del estudio.
Limitaciones del Estudio: Menciona las limitaciones del estudio y cómo podrían afectar los resultados.
Recomendaciones para Futuras Investigaciones: Proporciona recomendaciones para futuras investigaciones basadas en los hallazgos y limitaciones del estudio.
Fechas Relevantes: Incluye cualquier fecha futura relevante mencionada en las recomendaciones."""

# Configuraciones de generación y seguridad
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Función para extraer texto de un PDF con números de página
def extract_text_with_page_numbers(pdf_path):
    try:
        pdf_document = fitz.open("pdf", pdf_path.read())
        text_paginated = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text_paginated += f"--- Página {page_num + 1} ---\n"
            text_paginated += page.get_text()
            text_paginated += "\n\n"
        return text_paginated
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
        return None

# Función para generar contenido usando la API de Google Cloud
def multiturn_generate_content(text, container):
    model = GenerativeModel(
        "gemini-1.5-pro-001",
        system_instruction=[textsi_1]
    )
    chat = model.start_chat(response_validation= False)
    full_summary = ""
    for response in chat.send_message(text, generation_config=generation_config, safety_settings=safety_settings, stream=True):
        full_summary += response.candidates[0].content.parts[0].text
        container.markdown(full_summary, unsafe_allow_html=True)
    return full_summary

def main():
    st.title("Convertidor de PDF a Texto y Generador de Resúmenes con AI")
    
    uploaded_file = st.file_uploader("Sube un archivo PDF aquí", type="pdf")
      # Creamos un único contenedor para todos los resúmenes
    
    if uploaded_file is not None:
        text = extract_text_with_page_numbers(uploaded_file)
        
        if text:
            st.success("El PDF ha sido convertido exitosamente.")
            select_encoding = st.selectbox("Selecciona la codificación del texto", ["utf-8", "latin-1", "windows-1252"])
            output_path = uploaded_file.name.replace(".pdf", "_output.txt")
            with open(output_path, "w", encoding=select_encoding, errors="replace") as output_file:
                    output_file.write(text)
            
            st.session_state.show_ai_button = True
            st.session_state.show_save_button = True
            if st.session_state.show_ai_button:
              ai_button = st.button("Generar Resumen con AI" , key = "generate_summary")
            if st.session_state.show_save_button:
              save_button = st.download_button("Guardar Texto Extraído", text, f"{uploaded_file.name.replace('.pdf', '_output.txt')}", key = "save_text", mime="text/plain")
            if ai_button:
               st.session_state.show_success = False
               st.session_state.summary_container = st.empty()
               summary = multiturn_generate_content(text,  st.session_state.summary_container )
      
               user_input = st.text_area("¿Quieres ajustar algo en el resumen?", height=100)
               if user_input:
                   adjusted_input = "Instrucción adicional del usuario: " + user_input + "\n\nResumen generado previamente:\n" + summary +  "\n\nTexto original del documento:\n" + text
                   adjusted_summary = multiturn_generate_content(adjusted_input,  st.session_state.summary_container )

            elif save_button :
                st.session_state.show_success = True
                if st.session_state.show_success:
                  success = st.success("El texto ha sido guardado exitosamente.")
                

            

if __name__ == "__main__":
    main()
