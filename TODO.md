# Training metrics:

* La grafica de tokens procesados no esta bien porque muestra unos 80M tokens y son 475 millones, quizas no tiene en cuenta correctamente el gradient_accumulation

# Training components

* Nuestra implementacion de attention es purametne educativa, deberia tener un modulo de flash_attention mas "profesional", tomando como inspiracion a nanochat


# Tokenizer: 
* Hacer un fork de rust-bpe que permita mostrar el progreso y steps
* Evaluar pipeline streaming sin cache tokenizada de HF (workers -> cola en memoria acotada -> writer directo a shards .bin + manifest)

# swallow-code
Usar memmap para quedarnos solo con X tokens de training e Y tokens de validacion.

como sabemos el tipo podemos iterar por el archivo y pista