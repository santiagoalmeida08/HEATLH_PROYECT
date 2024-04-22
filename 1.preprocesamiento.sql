--1. La variable age convertirla a categoria con ordinal encoding

DROP TABLE IF EXISTS tabla1;
CREATE TABLE tabla1 AS SELECT * FROM hr;

-- Crear una tabla temporal para almacenar las categorías únicas junto con su codificación ordinal
DROP TABLE IF EXISTS categorias_unicas;
CREATE TABLE categorias_unicas AS
SELECT DISTINCT age
FROM tabla1;

-- Agregar una columna de codificación ordinal a la tabla temporal
ALTER TABLE categorias_unicas
ADD COLUMN cod_ordinal INTEGER;

-- Asignar números secuenciales a cada categoría
UPDATE categorias_unicas
SET cod_ordinal = (SELECT ROW_NUMBER()  OVER (ORDER BY age)
                    FROM categorias_unicas AS cu2
                    WHERE cu2.age = categorias_unicas.age);

ALTER TABLE tabla1 ADD COLUMN edad INTEGER;


-- Actualizar la tabla original con la codificación ordinal
UPDATE tabla1
SET categoria_encoded = (
    SELECT cod_ordinal 
    FROM categorias_unicas 
    WHERE categorias_unicas.age = tabla1.age);

-- Eliminar la tabla temporal
--DROP TABLE categorias_unicas;


DROP TABLE IF EXISTS hrmin;
CREATE TABLE hrmin AS 
SELECT * FROM hr;

UPDATE hrmin
SET medical_specialty = LOWER(medical_specialty),
    diag_1 = LOWER(diag_1),
    diag_2 = LOWER(diag_2),
    diag_3 = LOWER(diag_3);

UPDATE hrmin
SET medical_specialty = 
CASE 
    WHEN medical_specialty = 'emergency/trauma' THEN 'internalmedicine'
    ELSE medical_specialty
END;

UPDATE hrmin
SET diag_1 =
    CASE diag_1
        WHEN 'injury' THEN 'musculoskeletal'
        ELSE diag_1
    END,
    diag_2 =
    CASE diag_2
        WHEN 'injury' THEN 'musculoskeletal'
        ELSE diag_2
    END,
    diag_3 =
    CASE diag_3
        WHEN 'injury' THEN 'musculoskeletal'
        ELSE diag_3
    END;