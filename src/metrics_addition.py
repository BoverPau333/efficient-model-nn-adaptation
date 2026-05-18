"""Metricas reutilizables para estudios de adicion de clases.

Este modulo define las metricas mas utiles para comparar distintas maneras de
anadir una nueva clase y readaptar el modelo de la forma mas rapida posible,
manteniendo una buena accuracy tanto en la clase incorporada como en las clases
que el modelo ya conocia.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricaAdicion:
    """Representa una metrica relevante para metodos de adicion."""

    nombre: str
    nm: str
    prioridad: str
    por_que_importa: str


METRICAS_ADICION = [
    MetricaAdicion(
        nombre="Tiempo total de adaptacion",
        nm="t_total",
        prioridad="critica",
        por_que_importa=(
            "Es la metrica principal de eficiencia. Permite comparar de forma "
            "directa que metodo incorpora una clase nueva y readapta el modelo "
            "mas rapido."
        ),
    ),
    MetricaAdicion(
        nombre="Accuracy global",
        nm="acc_total",
        prioridad="critica",
        por_que_importa=(
            "Es la referencia principal de calidad final. Permite comprobar si "
            "la incorporacion de la nueva clase mantiene un buen rendimiento en "
            "el problema completo."
        ),
    ),
    MetricaAdicion(
        nombre="Accuracy por clase",
        nm="acc_pc",
        prioridad="alta",
        por_que_importa=(
            "Permite detectar si el metodo favorece la clase anadida a costa de "
            "degradar clases concretas que el modelo ya conocia."
        ),
    ),
    MetricaAdicion(
        nombre="Accuracy en clases previas",
        nm="acc_prev",
        prioridad="alta",
        por_que_importa=(
            "Mide hasta que punto el metodo conserva el conocimiento valido "
            "sobre las clases originales despues de incorporar una nueva."
        ),
    ),
    MetricaAdicion(
        nombre="Accuracy en la clase anadida",
        nm="acc_new",
        prioridad="critica",
        por_que_importa=(
            "Es la metrica mas directa para saber si la nueva clase se ha "
            "aprendido de verdad y no solo se ha anadido nominalmente al modelo."
        ),
    ),
    MetricaAdicion(
        nombre="Recall en la clase anadida",
        nm="rec_new",
        prioridad="alta",
        por_que_importa=(
            "Ayuda a medir cuantas muestras reales de la nueva clase consigue "
            "recuperar el modelo. Es clave cuando interesa no dejar ejemplos "
            "nuevos sin detectar."
        ),
    ),
    MetricaAdicion(
        nombre="Precision en la clase anadida",
        nm="prec_new",
        prioridad="alta",
        por_que_importa=(
            "Permite ver si el modelo confunde otras clases antiguas con la "
            "nueva clase, algo habitual cuando la incorporacion no queda bien "
            "separada."
        ),
    ),
    MetricaAdicion(
        nombre="F1 de la clase anadida",
        nm="f1_new",
        prioridad="alta",
        por_que_importa=(
            "Resume en una sola medida el equilibrio entre precision y recall "
            "sobre la clase nueva, especialmente util si hay desbalanceo."
        ),
    ),
    MetricaAdicion(
        nombre="Forgetting u olvido sobre clases previas",
        nm="olv_prev",
        prioridad="alta",
        por_que_importa=(
            "Cuantifica cuanto empeora el rendimiento en las clases ya "
            "aprendidas. Es clave para comparar rapidez de adicion frente a "
            "preservacion del conocimiento previo."
        ),
    ),
    MetricaAdicion(
        nombre="Numero de ejemplos utilizados",
        nm="n_ejmp",
        prioridad="media",
        por_que_importa=(
            "Permite comparar cuantas muestras necesita cada metodo para "
            "incorporar la nueva clase. Menos ejemplos suele implicar menor "
            "coste y menor tiempo."
        ),
    ),
    MetricaAdicion(
        nombre="Numero de ejemplos de la clase anadida",
        nm="n_ejmp_new",
        prioridad="media",
        por_que_importa=(
            "Hace visible la eficiencia especifica sobre la nueva clase y ayuda "
            "a comparar metodos en escenarios de pocos ejemplos o few-shot."
        ),
    ),
    MetricaAdicion(
        nombre="Confianza de prediccion",
        nm="conf",
        prioridad="media",
        por_que_importa=(
            "Ayuda a ver si el metodo mantiene predicciones seguras tras la "
            "adicion o si el modelo queda mas dubitativo aunque siga acertando."
        ),
    ),
    MetricaAdicion(
        nombre="Confianza media en la clase anadida",
        nm="conf_new",
        prioridad="media",
        por_que_importa=(
            "Permite comprobar si el modelo reconoce la nueva clase con "
            "seguridad o si la clasifica correctamente pero con alta "
            "incertidumbre."
        ),
    ),
    MetricaAdicion(
        nombre="Numero de parametros entrenados o modificados",
        nm="n_param",
        prioridad="media",
        por_que_importa=(
            "Es util para comparar metodos como reentrenamiento completo, "
            "fine-tuning parcial, LoRA o actualizacion solo de la cabecera."
        ),
    ),
    MetricaAdicion(
        nombre="Memoria adicional requerida",
        nm="ext_mem",
        prioridad="media",
        por_que_importa=(
            "Complementa al tiempo de adaptacion, porque algunos metodos pueden "
            "ser rapidos pero necesitar memoria extra para adaptadores, buffers "
            "o muestras almacenadas."
        ),
    ),
]


def imprimir_metricas_adicion():
    """Muestra un resumen legible de las metricas seleccionadas."""
    print("Metricas para comparar metodos de adicion:\n")
    for metrica in METRICAS_ADICION:
        print(f"- {metrica.nombre} [{metrica.prioridad}]")
        print(f"  {metrica.por_que_importa}\n")


if __name__ == "__main__":
    imprimir_metricas_adicion()
