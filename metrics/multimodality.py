import torch

class Multimodality:
    def compute_multimodality(self, generated_motions, text_prompts):
        """
        Computes multimodality score.
        Args:
            generated_motions (torch.Tensor): Motion sequences.
            text_prompts (list): Corresponding text prompts.
        Returns:
            float: Multimodality score.
        """
        motion_groups = {prompt: [] for prompt in set(text_prompts)}
        for i, prompt in enumerate(text_prompts):
            motion_groups[prompt].append(generated_motions[i])

        mm_scores = []
        for group in motion_groups.values():
            group_tensor = torch.stack(group)
            distances = torch.cdist(group_tensor, group_tensor, p=2)
            mm_scores.append(torch.mean(distances).item())

        return sum(mm_scores) / len(mm_scores)
