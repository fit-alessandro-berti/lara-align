import sys
from pathlib import Path

from pm4py.objects.log.obj import Event, Trace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lara_align import CertifyingAlignmentSystem, LARAMode
from lara_align.synthetic import make_sequence_net


def main() -> None:
    net, initial_marking, final_marking = make_sequence_net(["A", "B", "C"])
    trace = Trace(
        [
            Event({"concept:name": "A"}),
            Event({"concept:name": "X"}),
            Event({"concept:name": "B"}),
            Event({"concept:name": "C"}),
        ]
    )

    lara = CertifyingAlignmentSystem()
    result = lara.align(
        net,
        initial_marking,
        final_marking,
        trace,
        mode=LARAMode.CERTIFIED,
    )

    print("certified:", result.certified_optimal)
    print("cost:", result.cost)
    print("alignment:", result.alignment.to_pm4py_label_alignment())


if __name__ == "__main__":
    main()
