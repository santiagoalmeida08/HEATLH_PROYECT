
DROP TABLE IF EXISTS hrmin;
CREATE TABLE hrmin AS 
SELECT time_in_hospital, n_lab_procedures, n_procedures,	n_medications, 
n_outpatient, n_inpatient, n_emergency,	medical_specialty,	diag_1,
diag_2,	diag_3,	glucose_test,	A1Ctest, change, diabetes_med,	readmitted, edad FROM basecambios;



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
DROP TABLE IF EXISTS categorias_unicas;


