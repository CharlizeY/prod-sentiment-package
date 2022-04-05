import sparknlp
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)

    spark = sparknlp.start()
    
    print(f"Running Apache Spark version {spark.version}")
    print(f"Running JSL Spark NLP version {sparknlp.version()}")