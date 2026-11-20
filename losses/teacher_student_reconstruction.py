from .feature_norm import l2_per_sample


def teacher_student_reconstruction(student, reference):
    return l2_per_sample(student - reference).mean()
