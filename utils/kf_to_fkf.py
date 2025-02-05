import torch
from fuzzy_vocab import FuzzyLinguisticVocabulary

class KFtoFKFConverter:
    """
    Converts crisp Kinematic Facts (KFs) into Fuzzy Kinematic Facts (FKFs)
    using the fuzzy linguistic vocabulary.
    """

    def __init__(self):
        """
        Initializes fuzzy linguistic vocabulary.
        """
        self.fuzzy_vocab = FuzzyLinguisticVocabulary()

    def convert_kf_to_fkf(self, kinematic_facts):
        """
        Converts Kinematic Facts (KFs) into Fuzzy Kinematic Facts (FKFs).
        Args:
            kinematic_facts (dict): Dictionary of extracted KFs with crisp values.
        Returns:
            dict: FKFs with fuzzy linguistic labels.
        """
        fkfs = {}

        for key, value in kinematic_facts.items():
            # Determine category based on KF type
            if "distance" in key or "JJD" in key or "JLD" in key or "LLD" in key:
                category = "distance"
            elif "angle" in key or "JJA" in key or "JLA" in key or "LLA" in key:
                category = "angle"
            elif "speed" in key or "JS" in key or "LS" in key:
                category = "speed"
            else:
                continue  # Skip unrecognized categories

            # Convert crisp value to fuzzy term
            fuzzy_term, membership = self.fuzzy_vocab.get_fuzzy_term(torch.tensor(value), category)
            fkfs[key] = {"term": fuzzy_term, "membership": membership}

        return fkfs



if __name__ == "__main__":
    # Sample kinematic facts extracted from motion data
    sample_kfs = {
        "JJD": 0.35,  # Joint-Joint Distance
        "JLA": 45,  # Joint-Limb Angle
        "JS": 1.2,  # Joint Speed
        "LLA": -15,  # Limb-Limb Angle
    }

    converter = KFtoFKFConverter()
    fkfs = converter.convert_kf_to_fkf(sample_kfs)

    print("\nConverted FKFs:")
    for key, value in fkfs.items():
        print(f"{key}: {value['term']} (Membership: {value['membership']})")
