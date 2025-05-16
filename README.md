# Protein Design and Folding Stability Prediction

## Introduction

In the field of generative protein modeling, optimizing models for designing proteins that are both well-folded and stable is crucial. However, it remains unclear under which conditions either sequence or structure information, or both, is more important for generating stable proteins.

In our protein design approach, we have developed a joint model that combines structure and sequence models. We utilize TrRosetta to model the $p(structure|sequence)$ and train a new model TrMRF to model the $p(sequence|structure)$. Experimental validation shows the joint model achieves the most stable proteins.

## Key Features

- **Joint Model Development**: We combine structure and sequence models using TrORS and TrMRF respectively.
- **Experimental Validation**: Our joint model is validated through experimental results showcasing stability in designed proteins.
- **Hybrid Scoring Function**: We evaluate recent sequence and structure-based models and develop a hybrid scoring function integrating ESM2, ESMFold, and ProteinMPNN. This scoring function accurately predicts folding stability, surpassing individual models' accuracy.

  
## Usage
1. **Installation**: Clone the repository.
    ```bash
    chmod +x setup.sh
    ```
2. **Protein generation Model**: Utilize TrORS/TrMRF/(TrORS+TrMR) for protein generation
3. **Folding Stability**: Implement the hybrid scoring function for folding stability prediction.
4. **Pairwise Potentials**: Check sitewise/pairwise potentials of generated proteins.

## Future Work

- **Enhanced Protein Design**: Further optimization of joint models for better stability and functionality.
- **Scalability**: Extend the approach to larger protein design spaces.


## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

We acknowledge the support from MIT for this research.

## Contact

For any questions or inquiries, please contact Yehlin Cho at yehlin@mit.edu
