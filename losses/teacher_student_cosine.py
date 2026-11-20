import torch.nn.functional as F


def teacher_student_cosine(student, reference):
    similarity = F.cosine_similarity(student.float().flatten(1), reference.flatten(1), dim=1)
    return (1 - similarity).mean()
