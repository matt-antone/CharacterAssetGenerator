from cag.fidelity import BONES, angle_gap, bone_errors

POSE = {
    "shL": [1.1, 0.25], "shR": [0.9, 0.25], "elL": [1.2, 0.4], "elR": [0.8, 0.4],
    "wrL": [1.2, 0.55], "wrR": [0.8, 0.55], "hipL": [1.05, 0.55], "hipR": [0.95, 0.55],
    "knL": [1.05, 0.75], "knR": [0.95, 0.75], "anL": [1.05, 0.95], "anR": [0.95, 0.95],
}


def scaled(pose, factor, shift=0.3):
    return {k: [x * factor + shift, y * factor] for k, (x, y) in pose.items()}


def test_same_pose_at_another_size_and_place_scores_zero():
    """Proportion and framing are not pose: a bigger figure elsewhere is the same dance."""
    errors = bone_errors(scaled(POSE, 1.7), POSE)
    assert max(errors.values()) < 1e-9


def test_one_bent_forearm_is_the_only_error_and_its_size_is_the_bend():
    bent = dict(POSE, wrL=[1.35, 0.4])  # forearm turned from straight down to straight out
    errors = bone_errors(bent, POSE)
    assert round(errors["forearm L"]) == 90
    assert all(e < 1e-9 for name, e in errors.items() if name != "forearm L")
    assert set(errors) == set(BONES)


def test_angle_gap_wraps_around():
    assert angle_gap(179, -179) == 2
    assert angle_gap(-90, 90) == 180
