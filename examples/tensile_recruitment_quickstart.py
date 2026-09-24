"""Library-only example: python -m examples.tensile_recruitment_quickstart."""
import numpy as np

from fibernet.analysis import analyze_tensile_recruitment


class RecruitmentExample:
    def run(self):
        edges = np.array([[0, 1], [1, 2], [0, 2]])
        edge_strain = np.array([[0.0, 0.0, 0.0],
                                [0.2, 0.3, -0.1],
                                [0.02, 0.3, 0.0]])
        result = analyze_tensile_recruitment(
            edge_strain, edges, left_nodes=[0], right_nodes=[2],
            n_nodes=3, alpha=0.1, hysteresis=0.5)
        assert result.perc_frame == 1
        assert result.active_edges[-1].tolist() == [True, True, False]
        print('first spanning frame:', result.perc_frame)
        print('recruited edge fraction:', result.active_edges.mean(axis=1))
        return result


if __name__ == '__main__':
    RecruitmentExample().run()
