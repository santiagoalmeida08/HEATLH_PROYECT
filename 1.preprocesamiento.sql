--1. La variable age convertirla a categoria con ordinal encoding

DROP TABLE IF EXISTS tabla1;
CREATE TABLE tabla1 AS SELECT * FROM hr;

UPDATE tabla1
SET age=
CASE 
    WHEN '[50-60)' THEN 1
    WHEN '[60-70)' THEN 2
    WHEN '[70-80)' THEN 3
    WHEN '[80-90)' THEN 4
    WHEN '[90-100)' THEN 5
    ELSE age 
END;


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

UPDATE hrim 
SET age=
    CASE
    WHEN '[50-60)' THEN 1
    WHEN '[60-70)' THEN 2
    WHEN '[70-80)' THEN 3
    WHEN '[80-90)' THEN 4
    WHEN '[90-100)' THEN 5
    ELSE age 
END;