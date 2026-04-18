"""Metricas reutilizables para estudios de eliminacion de clases.

Este modulo define las metricas mas utiles para comparar distintas maneras de
eliminar una clase y readaptar el modelo de la forma mas rapida posible,
manteniendo una buena accuracy en las clases restantes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricaEliminacion:
    """Representa una metrica relevante para comparar metodos de eliminacion."""

    nombre: str
    prioridad: str
    por_que_importa: str


METRICAS_ELIMINACION = [
    MetricaEliminacion(
        nombre="Tiempo total de adaptacion",
        prioridad="critica",
        por_que_importa=(
            "Es la metrica principal de eficiencia. Permite comparar de forma "
            "directa que metodo elimina una clase y readapta el modelo mas rapido."
        ),
    ),
    MetricaEliminacion(
        nombre="Accuracy global",
        prioridad="critica",
        por_que_importa=(
            "Es la referencia principal de calidad final. Permite comprobar si la "
            "ganancia en tiempo mantiene un buen rendimiento global."
        ),
    ),
    MetricaEliminacion(
        nombre="Accuracy por clase",
        prioridad="alta",
        por_que_importa=(
            "Permite detectar si algunos metodos perjudican mas que otros a "
            "determinadas clases restantes tras la eliminacion."
        ),
    ),
    MetricaEliminacion(
        nombre="Accuracy en clases restantes",
        prioridad="alta",
        por_que_importa=(
            "Mide hasta que punto cada metodo conserva el conocimiento valido "
            "sobre las clases que siguen existiendo despues de borrar una clase."
        ),
    ),
    MetricaEliminacion(
        nombre="Forgetting u olvido",
        prioridad="alta",
        por_que_importa=(
            "Cuantifica cuanto empeora el rendimiento en las clases que se "
            "mantienen tras la readaptacion. Es clave para comparar rapidez "
            "frente a perdida de conocimiento previo."
        ),
    ),
    MetricaEliminacion(
        nombre="Numero de ejemplos utilizados",
        prioridad="media",
        por_que_importa=(
            "Permite comparar cuantas muestras necesita cada metodo para "
            "readaptarse despues de eliminar una clase. Menos ejemplos suele "
            "implicar menor coste y menor tiempo."
        ),
    ),
    MetricaEliminacion(
        nombre="Confianza de prediccion",
        prioridad="media",
        por_que_importa=(
            "Ayuda a ver si un metodo mantiene predicciones seguras despues de "
            "la eliminacion o si, aunque acierte, lo hace con mas incertidumbre."
        ),
    ),
    MetricaEliminacion(
        nombre="Numero de parametros entrenados o modificados",
        prioridad="media",
        por_que_importa=(
            "Es util para comparar metodos como reentrenamiento completo, "
            "fine-tuning parcial, LoRA o actualizacion solo de la cabecera."
        ),
    ),
    MetricaEliminacion(
        nombre="Memoria adicional requerida",
        prioridad="media",
        por_que_importa=(
            "Complementa al tiempo de adaptacion, porque algunos metodos pueden "
            "ser rapidos pero necesitar memoria extra para adaptadores, buffers "
            "o muestras almacenadas."
        ),
    ),
]


METRICAS_MENOS_RELEVANTES_O_NO_APLICABLES = [
    {
        "nombre": "Accuracy en clases nuevas o modificadas",
        "motivo": (
            "No suele aplicar directamente a eliminacion de clases, porque no se "
            "anaden clases nuevas. Solo seria util si ademas se redefine alguna "
            "clase existente."
        ),
    },
]


def imprimir_metricas_eliminacion():
    """Muestra un resumen legible de las metricas seleccionadas."""
    print("Metricas recomendadas para comparar metodos de eliminacion:\n")
    for metrica in METRICAS_ELIMINACION:
        print(f"- {metrica.nombre} [{metrica.prioridad}]")
        print(f"  {metrica.por_que_importa}\n")

    if METRICAS_MENOS_RELEVANTES_O_NO_APLICABLES:
        print("Metricas menos relevantes o no aplicables:\n")
        for item in METRICAS_MENOS_RELEVANTES_O_NO_APLICABLES:
            print(f"- {item['nombre']}")
            print(f"  {item['motivo']}\n")


if __name__ == "__main__":
    imprimir_metricas_eliminacion()
