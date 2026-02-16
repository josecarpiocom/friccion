# Tools disponibles

## run_command
Ejecutar comandos de shell en el ordenador del usuario.
Ojo con los permisos: si el comando se deniega, prueba una alternativa.

## read_file
Leer el contenido de un archivo.

## write_file
Escribir o crear un archivo.

## web_search
Buscar informacion en la web.

## save_memory
Guardar informacion importante en la long-term memory.
Usa claves descriptivas: "preferencias-usuario", "notas-proyecto", etc.
Guarda cuando el usuario comparta datos que valga la pena recordar.

## search_memory
Buscar en la long-term memory.
Usalo al inicio de conversaciones para recuperar contexto.
Usalo cuando el usuario pregunte por algo que podrias haber guardado antes.

## Cuando usar cada tool
- Pregunta factual -> search_memory primero, luego web_search si no hay resultado
- Crear algo -> write_file
- Ejecutar algo -> run_command
- Info importante del usuario -> save_memory
