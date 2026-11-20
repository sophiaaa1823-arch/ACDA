import torch

from .filtered_feature_consistency import filtered_feature_consistency
from .teacher_student_cosine import teacher_student_cosine
from .teacher_student_reconstruction import teacher_student_reconstruction


def local_alignment(teacher, decoded, filtered, weights):
    cosine, reconstruction, consistency = [], [], []
    for reference, student, skip in zip(teacher, decoded, filtered):
        reference = torch.cat((reference, reference)).float()
        cosine.append(teacher_student_cosine(student, reference))
        reconstruction.append(teacher_student_reconstruction(student, reference))
        clean, perturbed = skip.chunk(2)
        consistency.append(filtered_feature_consistency(clean, perturbed))
    return (
        weights["cosine"] * sum(cosine)
        + weights["reconstruction"] * sum(reconstruction)
        + weights["consistency"] * sum(consistency)
    )
