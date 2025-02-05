import torch

class KinematicFactsExtractor:
    """
    Extracts kinematic facts (KFs) from raw motion sequences.
    """

    def __init__(self, num_joints=22):
        """
        Args:
            num_joints (int): Number of joints in the motion sequence.
        """
        self.num_joints = num_joints

    def compute_joint_joint_angle(self, joint_positions):
        """
        Computes the angle between two connected joints.
        Args:
            joint_positions (torch.Tensor): Joint positions (batch_size, num_joints, 3).
        Returns:
            torch.Tensor: Joint-Joint Angle matrix.
        """
        diff_vectors = joint_positions[:, :, None, :] - joint_positions[:, None, :, :]
        norms = torch.norm(diff_vectors, dim=-1, keepdim=True)
        unit_vectors = diff_vectors / (norms + 1e-6)  # Normalize to unit vectors
        cos_theta = torch.sum(unit_vectors[:, :, None, :] * unit_vectors[:, None, :, :], dim=-1)
        angles = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
        return angles

    def compute_joint_joint_distance(self, joint_positions):
        """
        Computes Euclidean distance between all pairs of joints.
        Args:
            joint_positions (torch.Tensor): Joint positions (batch_size, num_joints, 3).
        Returns:
            torch.Tensor: Joint-Joint Distance matrix.
        """
        distances = torch.norm(joint_positions[:, :, None, :] - joint_positions[:, None, :, :], dim=-1)
        return distances

    def compute_joint_limb_distance(self, joint_positions):
        """
        Computes distance between a joint and the midpoint of a limb.
        Args:
            joint_positions (torch.Tensor): Joint positions (batch_size, num_joints, 3).
        Returns:
            torch.Tensor: Joint-Limb Distance matrix.
        """
        midpoints = (joint_positions[:, :-1, :] + joint_positions[:, 1:, :]) / 2
        distances = torch.norm(joint_positions[:, :, None, :] - midpoints[:, None, :, :], dim=-1)
        return distances

    def compute_joint_displacement(self, joint_positions):
        """
        Computes the displacement of joints between consecutive frames.
        Args:
            joint_positions (torch.Tensor): Joint positions (batch_size, time_steps, num_joints, 3).
        Returns:
            torch.Tensor: Joint Displacement matrix.
        """
        displacements = torch.norm(joint_positions[:, 1:, :, :] - joint_positions[:, :-1, :, :], dim=-1)
        return displacements

    def compute_joint_speed(self, joint_positions, fps=30):
        """
        Computes the speed of joints between consecutive frames.
        Args:
            joint_positions (torch.Tensor): Joint positions (batch_size, time_steps, num_joints, 3).
            fps (int): Frames per second.
        Returns:
            torch.Tensor: Joint Speed matrix.
        """
        displacements = self.compute_joint_displacement(joint_positions)
        speed = displacements * fps  # Convert displacement to velocity
        return speed

    def compute_limb_speed(self, joint_positions, fps=30):
        """
        Computes speed of limbs by averaging joint speeds.
        Args:
            joint_positions (torch.Tensor): Joint positions (batch_size, time_steps, num_joints, 3).
            fps (int): Frames per second.
        Returns:
            torch.Tensor: Limb Speed matrix.
        """
        speeds = self.compute_joint_speed(joint_positions, fps)
        limb_speeds = (speeds[:, :, :-1] + speeds[:, :, 1:]) / 2
        return limb_speeds

    def extract_kinematic_facts(self, joint_positions):
        """
        Extracts all kinematic facts from raw motion sequences.
        Args:
            joint_positions (torch.Tensor): Joint positions (batch_size, time_steps, num_joints, 3).
        Returns:
            dict: Dictionary of extracted kinematic facts.
        """
        kinematic_facts = {
            "JJA": self.compute_joint_joint_angle(joint_positions[:, -1, :, :]),  # Last frame
            "JJD": self.compute_joint_joint_distance(joint_positions[:, -1, :, :]),
            "JLD": self.compute_joint_limb_distance(joint_positions[:, -1, :, :]),
            "JD": self.compute_joint_displacement(joint_positions),
            "JS": self.compute_joint_speed(joint_positions),
            "LS": self.compute_limb_speed(joint_positions),
        }
        return kinematic_facts




if __name__ == "__main__":
    batch_size = 16
    time_steps = 60  # Example motion sequence length
    num_joints = 22  # Number of body joints
    joint_positions = torch.randn(batch_size, time_steps, num_joints, 3)  # Simulated motion data

    extractor = KinematicFactsExtractor(num_joints)
    kinematic_facts = extractor.extract_kinematic_facts(joint_positions)

    for key, value in kinematic_facts.items():
        print(f"{key}: {value.shape}")  # Prints shape of each kinematic fact
