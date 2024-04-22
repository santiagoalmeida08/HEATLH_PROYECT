
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


DROP TABLE IF EXISTS tabla1;
CREATE TABLE tabla1 AS 
SELECT * FROM re2;

ALTER TABLE tabla1 drop column age;