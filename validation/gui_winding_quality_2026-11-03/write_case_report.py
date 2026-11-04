"""Join matched contour and final-wire measurements for the current gallery."""

import csv
from pathlib import Path

from coilgen.application.templates import TEMPLATE_SPECS, template_assembly


HERE = Path(__file__).resolve().parent
VERDICTS = {"True": "pass", "False": "fail"}


def read_table(filename, key):
    with (HERE / filename).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {key(row): row for row in rows}
    if len(rows) != len(indexed):
        raise ValueError(f"Duplicate case in {filename}")
    return indexed


def main():
    contours = read_table("gallery.csv", lambda row: row["case"])
    mounts = read_table("full-gallery.csv", lambda row: f"{row['template']}__{row['channel_index']}")
    expected = {
        f"{spec.id}__{i}": (spec, channel.project)
        for spec in TEMPLATE_SPECS
        for i, channel in enumerate(template_assembly(spec.id).channels)
    }
    if expected.keys() != contours.keys() or expected.keys() != mounts.keys():
        raise ValueError("The reports must cover exactly the channels in the current gallery.")
    counts = dict(
        accepted=0, local=0, total=0, angle_improved=0, angle_unchanged=0, angle_worse=0, sheet=0, sweep=0, fab=0
    )
    sections = []
    group = None
    for key, (spec, project) in expected.items():
        c, w = contours[key], mounts[key]
        if (
            c["model_hash"] != project.model_fingerprint
            or w["model_fingerprint"] != project.model_fingerprint
            or w["project_fingerprint"] != project.fingerprint
        ):
            raise ValueError(f"Outdated measurement: {key}")
        if int(c["accepted"]) != int(w["smoothed"]) or int(c["local_fits"]) != int(w["local"]):
            raise ValueError(f"Paired fit and full pipeline disagree: {key}")
        count = int(c["turns"])
        if count != int(w["smoothed"]) + int(w["original_retained"]):
            raise ValueError(f"Incomplete fit accounting: {key}")
        if group != spec.category:
            group = spec.category
            sections.extend(
                [
                    f"\n## {group}\n",
                    "| Channel | Fits / loops (local) | Before routing: P95 ° | Mounted: P95 ° | Mounted: Bz RMS % | Min reviewed bend, mm | Sweep / fabrication |",
                    "|---|---:|---:|---:|---:|---:|---|",
                ]
            )
        deg0, deg1 = float(w["old_review_p95_deg"]), float(w["review_p95_deg"])
        counts["accepted"] += int(c["accepted"])
        counts["local"] += int(c["local_fits"])
        counts["total"] += count
        counts[
            "angle_improved" if deg1 < deg0 - 0.01 else "angle_worse" if deg1 > deg0 + 0.01 else "angle_unchanged"
        ] += 1
        counts["sheet"] += w["sheet_accepted"] == "True"
        counts["sweep"] += w["pipeline_sweep"] == "True"
        counts["fab"] += w["fabrication"] == "True"
        case_name = project.name if len(template_assembly(spec.id).channels) == 1 else f"{spec.title}: {project.name}"
        sweep = VERDICTS.get(w["pipeline_sweep"], "unrecorded")
        fabrication = VERDICTS.get(w["fabrication"], "unrecorded")
        sections.append(
            f"| {case_name} | {c['accepted']} / {count} ({c['local_fits']}) | "
            f"{float(c['old_p95_deg']):.2f} → {float(c['smooth_p95_deg']):.2f} | {deg0:.2f} → {deg1:.2f} | "
            f"{float(w['previous_review_rms_percent']):.3f} → {float(w['review_rms_percent']):.3f} | "
            f"{float(w['min_fabrication_bend_mm']):.3f} | {sweep} / {fabrication} |"
        )
    introduction = [
        "# Winding results by channel\n",
        f"The updated templates fitted **{counts['accepted']} of {counts['total']} extracted loops**, including {counts['local']} accepted local fits. Across {len(expected)} mounted windings, P95 turning at 3 mm spacing fell for {counts['angle_improved']}, changed by at most 0.01° for {counts['angle_unchanged']}, and rose for {counts['angle_worse']}. Pass counts: independent sheet {counts['sheet']}/{len(expected)}, pipeline sweep {counts['sweep']}/{len(expected)}, fabrication review {counts['fab']}/{len(expected)}.\n",
        "Each arrow compares the previous published gallery to the updated template at each run’s original current. Angles measure direction change at equal 3 mm arclength stations. Bz RMS is the relative design-DSV error of the mounted winding. A passing sheet does not qualify a conductor. The [before/after screenshot comparison](README.md) uses the user’s exact original job; it is distinct from the published reference used in this table.\n",
        "[Contours and model fingerprints](gallery.csv) · [Final winding metrics and geometry hashes](full-gallery.csv) · [Saved mounted wires](reviewed_centerlines/)",
    ]
    (HERE / "case-results.md").write_text("\n".join(introduction + sections) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
