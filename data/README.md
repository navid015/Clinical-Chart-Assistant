# Synthetic patient data

All six patients, their names, MRNs, clinicians and histories are fictional and were written for this project. Any resemblance to real people is coincidental. No real patient data is used anywhere in this repository.

Each patient folder contains:
- `patient.json`: structured data. Allergies, problems (ICD-10-CM), medications with start and stop dates and reasons, and lab results (LOINC codes, reference ranges).
- `notes/*.md`: clinical notes with front matter (`patient_id`, `doc_type`, `title`, `date`, `author`) and `##` section headings, which the chunker uses.

The clinical content was written to be internally consistent and plausible, but it has not been reviewed by a clinician and must not be used as medical reference material. Codes are included to demonstrate terminology handling and should be verified before any other use.

| ID | Patient | Main conditions |
|---|---|---|
| P001 | Evelyn Hartwell | Type 2 diabetes with CKD 3a, retinopathy, hypertension |
| P002 | Raymond Okafor | HFrEF, CAD with stent, paroxysmal atrial fibrillation |
| P003 | Priya Natarajan | Moderate persistent asthma, migraine |
| P004 | Dolores Kincaid | Hip fracture with hemiarthroplasty, COPD, osteoporosis |
| P005 | Marcus Bellweather | Stage IIIB colon cancer after surgery and CAPOX |
| P006 | Tomas Rivera | Hypothyroidism, depression, severe sleep apnea |

To add a patient, create `data/patients/P00N/` with the same structure. The index rebuilds automatically when data changes.
