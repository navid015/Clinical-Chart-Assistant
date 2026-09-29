from datetime import date


def test_all_patients_load(patients):
    assert len(patients) >= 5
    for pid, p in patients.items():
        assert p.id == pid
        assert p.notes, f"{pid} has no notes"
        assert p.allergies, f"{pid} must record allergies or NKDA explicitly"


def test_notes_have_sections_and_valid_dates(patients):
    for p in patients.values():
        for n in p.notes:
            assert n.sections, n.path
            date.fromisoformat(n.date)


def test_every_problem_has_icd10_and_every_lab_has_loinc(patients):
    for p in patients.values():
        assert all(x.get("icd10") for x in p.problems), p.id
        assert all(l.get("loinc") for l in p.labs), p.id


def test_lab_results_are_in_date_order(patients):
    for p in patients.values():
        for lab in p.labs:
            dates = [r["date"] for r in lab["results"]]
            assert dates == sorted(dates), f"{p.id} {lab['test']}"


def test_age_calculation(patients):
    p = next(iter(patients.values()))
    born = date.fromisoformat(p.dob)
    assert p.age(date(born.year + 30, born.month, born.day)) == 30
