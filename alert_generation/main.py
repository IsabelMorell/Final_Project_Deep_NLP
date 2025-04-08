import utils as u
import os

paths = [
    "Ejercicios/Ejercicio_1/enunciado.txt",  # [E] Enunciado
]

placeholder = [
    "[E]"
]

def sustituir_prompt(prompt: str, paths: list, placeholders: list) -> str:
    """
    Reemplaza los placeholders en el prompt con el contenido de los archivos en paths.
    
    prompt: str: Enunciado con los placeholders [E], [P], [R], [C], [A]
    paths: list: Lista de rutas a los archivos [Enunciado, Código Profesor, Rubrica, Correcion, Código Alumno]
    
    Returns:
    str: El prompt con los placeholders reemplazados por el contenido de los archivos.
    """

    # Loop through the placeholders and replace them in the prompt
    for i in range(len(placeholders)):
        # Read the content of each file
        with open(paths[i], 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Replace the placeholder with the file content
        prompt = prompt.replace(placeholders[i], content)
    
    return prompt

def abrir_y_ejecutar_prompt():
    # Leer el contenido del archivo 'prompt.txt' que está en la carpeta actual
    try:
        with open('prompt.txt', 'r', encoding='utf-8') as file:
            prompt = file.read()
    except FileNotFoundError:
        print("El archivo 'prompt.txt' no se encuentra en la carpeta actual.")
        return
    except Exception as e:
        print(f"Error al leer el archivo 'prompt.txt': {e}")
        return
    
    prompt_sustituido = sustituir_prompt(prompt, paths, placeholder)
    
    # Obtener los modelos disponibles
    models = u.get_available_models()
    if not models:
        print("No se encontraron modelos disponibles.")
        return
    
    # Para cada modelo, ejecutar la función 'prompt_model' pasando el prompt sustituido
    for model in models:
        u.delete_history()
        print(f"Ejecutando para el modelo: {model}")
        try:
            u.prompt_model(model, prompt_sustituido)
        except Exception as e:
            print(f"Error al ejecutar para el modelo {model}: {e}")

if __name__ == "__main__":
    abrir_y_ejecutar_prompt()