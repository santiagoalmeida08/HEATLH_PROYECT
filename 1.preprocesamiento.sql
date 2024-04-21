--1. La variable age convertirla a categoria con ordinal encoding

DROP TABLE IF EXISTS tabla1;
CREATE TABLE tabla1 AS SELECT * FROM hr;

-- Crear una tabla temporal para almacenar las categorías únicas junto con su codificación ordinal
CREATE TEMP TABLE categorias_unicas AS
SELECT DISTINCT age
FROM tabla1;

-- Agregar una columna de codificación ordinal a la tabla temporal
ALTER TABLE categorias_unicas
ADD COLUMN cod_ordinal INTEGER;

-- Asignar números secuenciales a cada categoría
UPDATE categorias_unicas
SET cod_ordinal = (SELECT COUNT(*) FROM categorias_unicas cu2 WHERE cu2.age <= categorias_unicas.age);

-- Actualizar la tabla original con la codificación ordinal
UPDATE tabla1
SET categoria_encoded = (
    SELECT cod_ordinal 
    FROM categorias_unicas 
    WHERE categorias_unicas.age = tabla1.age
);

-- Eliminar la tabla temporal
DROP TABLE categorias_unicas;