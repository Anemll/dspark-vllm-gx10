

# DSpark vLLM para dos nodos DGX Spark / ASUS GX10

Esta es una adaptación (port) probada a GB10 para dos nodos de la ruta de servicio DeepSeek V4 Flash DSpark/NVFP4 para vLLM 0.25.1. Conecta el entorno de ejecución DeepSeek V4 de vLLM con el kernel nativo sparse-MLA SM120/SM121 de FlashInfer, añade un backend MoE b12x nativo MXFP4, y empaqueta una implementación reproducible, un panel en vivo, cambio de versión y pruebas de rendimiento.

## Configuración validada

- 2 × NVIDIA DGX Spark o ASUS Ascent GX10 (GB10, SM121, ARM64)
- red de alta velocidad dedicada entre nodos
- paralelismo de tensores: TP=2
- Modelo DeepSeek V4 Flash DSpark que utiliza caché KV DS MLA NVFP4
- Etiqueta de origen de vLLM `v0.25.1`; el entorno de ejecución informa `0.25.2.dev0+g752a3a504.d20260714`
- FlashInfer fijado en `0472b9b3f2fba11b463f8526f390297d52a8aad7`
- b12x fijado en `7dc6fb8fcc6446ea093537d1657df81985fa5f43`

## Qué cambia esta adaptación

- añade `nvfp4_ds_mla` como un formato de caché KV DeepSeek V4 de primera clase en toda la configuración de vLLM, la cuantización y el cálculo del tamaño de caché;
- utiliza la envoltura de token sparse-MLA empaquetada de 584 bytes probada para ambos grupos de caché MLA y de ventana deslizante;
- adapta el contenedor FlashInfer SM120/SM121 de vLLM para dividir páginas SWA de 256 tokens de tamaño excesivo en vistas de 64 tokens sin copias, preservando las páginas C128 comprimidas;
- soporta los 32 cabezales de consulta de TP=2 y rellena los anchos de índice disperso no soportados hasta los anchos nativos de despacho 128/512/1024 de FlashInfer con centinelas de ranura inválida;
- añade un backend MoE b12x MXFP4 modular con preparación nativa de pesos, memoria temporal propiedad del llamante, ejecución segura para gráficos CUDA, ajuste de pequeña M para GB10 y calentamiento de especialización de paquete de rutas al inicio;
- añade herramientas Compose/iniciar/actualizar para dos nodos, junto con el panel en tiempo real independiente y el sistema de prueba de rendimiento controlado.

La implementación exacta a nivel de archivo se describe en [docs/implementation.md](docs/implementation.md).

Los pesos del modelo **no** están incluidos. Establezca `DSPARK_MODEL_HOST` en un directorio de modelo con licencia para su uso.

## Instalación

Clone este repositorio en ambos nodos:

```bash
git clone https://github.com/anemll/dspark-vllm-gx10.git
cd dspark-vllm-gx10
./scripts/install.sh --role worker
./scripts/install.sh --role head
```

Edite `config/worker.env` y `config/head.env`. Reemplace cada valor `CHANGEME`, verifique las rutas del modelo/caché y utilice las direcciones de la red dedicada, no Wi-Fi ni la LAN general.

Inicie el rango 1 primero, luego el rango 0:

```bash
# Worker
./scripts/start-node.sh config/worker.env

# Wait until rank 1 is listening for the rendezvous, then on the head:
./scripts/start-node.sh config/head.env
```

La API head está disponible en `http://HEAD_HOST:8888`. Un inicio exitoso devuelve HTTP 200 desde `/health` y la cadena del entorno de ejecución desde `/version`.

## Dashboard

El dashboard es un servicio Python sin dependencias. Muestra el rendimiento de decodificación/prefill, promedios activos no nulos, totales de tokens, aceptación DSpark, latencia de solicitudes, estado de carga para ambos rangos TP, versión de vLLM, temperatura, energía, utilización de GPU y temperatura NVMe opcional.

Consulte [docs/dashboard.md](docs/dashboard.md) para la instalación y configuración.

## Actualización

Actualice el repositorio y prepare una nueva etiqueta de imagen en cada nodo:

```bash
git pull --ff-only
./scripts/update.sh 0.1.1 config/worker.env
./scripts/update.sh 0.1.1 config/head.env
```

Reinicie el rango 1 primero y el rango 0 segundo. El actualizador conserva una copia de seguridad con marca de tiempo del archivo de entorno anterior.

## Compilar desde el código fuente

La imagen ARM64 precompilada se publica como:

```text
ghcr.io/anemll/dspark-vllm-gx10:0.1.1
```

Para reproducirla localmente, ejecute `./scripts/build-image.sh`. El script obtiene el commit exacto de vLLM en `upstream.lock`, aplica `overlay/`, compila la imagen ARM64 de vLLM e instala las revisiones fijas de FlashInfer y b12x de Git. El anclaje de b12x es intencionalmente un commit de Git: la fuente probada `0.15.3` no se publicó en PyPI.

`docker/Dockerfile.promote-tested` es un paso de lanzamiento exclusivo para mantenedores que añade etiquetas de fuente/versión OCI y avisos de licencia incluidos a una imagen que ya ha superado la validación de dos nodos. No reemplaza la compilación reproducible desde el código fuente anterior ni modifica el código del entorno de ejecución.

## Rendimiento

Modelo de referencia (benchmark):

- [drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored](https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored)
- DeepSeek V4 Flash, 284B MoE / aproximadamente 13B de parámetros activos
- Pesos FP8 (E4M3, escalas UE8M0, bloques 128x128): 48 fragmentos Safetensors que totalizan 166,886,535,336 bytes (155.43 GiB) en cada nodo
- caché KV `nvfp4_ds_mla`; NVFP4 describe el formato de caché DS-MLA, no el formato de peso FP8 del punto de control
- TP=2 en dos nodos GX10; contexto máximo del servidor 350,000 tokens

Referencia de nodo único: el punto de control sin cambios **no es ejecutable en un solo GX10**. Sus 155.43 GiB de archivos de peso superan los aproximadamente 121 GiB de memoria unificada utilizable del nodo antes de la caché KV y las asignaciones del entorno de ejecución. Un lanzamiento controlado TP=1 alcanzó `NV_ERR_NO_MEMORY` de NVIDIA antes de que la API estuviera lista, por lo que no hay muestras de rendimiento válidas de nodo único. El registro completo de [fit-check record](benchmarks/results/prefill-v0251-single-node-fit.md) se conserva junto con los resultados del benchmark.

Mejor rendimiento de salida agregado desde la carga de trabajo controlada de 512 tokens:

| Concurrencia | Entorno de ejecución anterior | Candidato vLLM 0.25 | Ganancia |
|---:|---:|---:|---:|
| 1 | 40.7 tok/s | 48.5 tok/s | 19.1% |
| 2 | 59.1 tok/s | 70.4 tok/s | 19.2% |
| 4 | 91.4 tok/s | 103.5 tok/s | 13.2% |

Los resultados brutos y el cliente sin dependencias están en `benchmarks/`.
La [validación de calentamiento de route-pack](benchmarks/results/route-pack-warmup-v025.md) post-corrección incluye cobertura de límites JIT estrictos, una verificación de prefill de 65K y resultados de regresión de decodificación.

Resultados de prefill del lado del servidor con calentamiento en la misma implementación TP=2 de dos nodos:

| Tokens de entrada | vLLM 0.21.1 | Candidato vLLM 0.25 | Ganancia |
|---:|---:|---:|---:|
| 1,024 | 1,778.7 tok/s | 2,033.0 tok/s | 14.3% |
| 2,048 | 1,990.5 tok/s | 2,252.0 tok/s | 13.1% |
| 4,096 | 2,083.1 tok/s | 2,320.7 tok/s | 11.4% |
| 8,192 | 2,049.8 tok/s | 2,184.2 tok/s | 6.6% |
| 16,384 | 2,052.6 tok/s | 2,203.8 tok/s | 7.4% |
| 32,768 | 1,901.1 tok/s | 2,176.1 tok/s | 14.5% |

La [comparación](benchmarks/results/prefill-v0211-vs-v0251.md) y los [informes brutos](benchmarks/results/) contienen detalles de TTFT y por prueba.

El prefill se mide con longitudes de entrada exactas de 1K, 2K, 4K, 8K, 16K y 32K.
El sistema registra el TTFT del cliente más la duración del prefill del lado del servidor de vLLM y el conteo de tokens calculados. Utiliza prompts de ID de token únicos y reproducibles para que las ejecuciones antes y después reciban la misma entrada idéntica sin reutilización de la caché de prefijo.
Un calentamiento inicial más un paso excluido en cada longitud de entrada probada evita que la compilación de la primera forma contamine las medianas de las tres pruebas:

```bash
# Run against the previous runtime, then switch the two-node server version.
python3 benchmarks/benchmark_prefill.py --label before \
  --output benchmarks/results/prefill-before.json

# Run the identical matrix against the candidate runtime.
python3 benchmarks/benchmark_prefill.py --label after \
  --output benchmarks/results/prefill-after.json

python3 benchmarks/compare_prefill.py \
  benchmarks/results/prefill-before.json \
  benchmarks/results/prefill-after.json
```

Ejecute estas pruebas en un servidor que esté inactivo de lo contrario. El sistema detecta solicitudes superpuestas y excluye pruebas del lado del servidor contaminadas de su mediana.

## Nota operativa importante

Inicie el worker antes del head. Iniciar ambos simultáneamente puede dejar la inicialización distribuida esperando en TCPStore/NCCL. Los scripts proporcionados no almacenan contraseñas ni credenciales SSH.

## Licencia y atribución

El dashboard local al repositorio, la implementación, la documentación y el trabajo de benchmark están licenciados bajo MIT en [LICENSE](LICENSE). Los archivos derivados de vLLM bajo `overlay/` permanecen bajo Apache-2.0; el texto completo de Apache se incluye en [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

Consulte [CREDITS.md](CREDITS.md) para las revisiones exactas de las dependencias y el crédito explícito a vLLM, FlashInfer, Luke Alonso/b12x, voipmonitor, Keys/drowzeys, Rafael Caricio, MiaAI-Lab, TonyD2Wild, Fraser Price y roady001. Los pesos del modelo no están incluidos ni se relicencian.
