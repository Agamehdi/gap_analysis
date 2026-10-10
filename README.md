# WI System Gap Analysis

Install and run from the folder containing the Python script:

```bash
python -m pip install -r requirements.txt
python -m streamlit run WI_System_Gap_Analysis.py
```

Upload your well-target CSV and facility.csv through the sidebar Browse buttons.
The included facility.csv preserves your earlier long table and applies your
updated CPS/manifold limits only for October–December 2026. CAP units are mbd.
Review it before using it for another forecast period.

The well CSV uses US dates (10/09/2026 means 9 October 2026).
Select a date to calculate a snapshot; optionally compare a second date.
The supplied CSV has one row per well, so actual historical changes before
those records cannot be reconstructed. Multiple dated records per well are
supported; the latest row on/before the selected date is used.

The reference-style table uses grouped headers, merged CPS cells and heatmap
gap colors. Flat tables, well audit, missing-capacity wells, daily forecast
results and system totals can be exported to Excel. CSV and HTML downloads
are also available.

Calculations use bbl/day internally and mbd in the gap table. For both Agreed
Target and RMT+OE: cap each well at its BMT constrained capacity, then cap each
manifold, then cap each CPS. Facility restrictions are allocated proportionally
to contributing wells. Gaps are negative incremental losses at each stage.
An additional efficiency factor is off by default, and can be applied after
the constraints if your target is not already adjusted. These rules are shown
in the app because the supplied screenshot does not expose its formulas.
The default displayed injection is the Agreed Target scenario. Facility
constraints are assumed to have the same pre-efficiency basis as BMT and RMT+OE.

Missing BMT values default to exclusion with a visible list (34 in your supplied
file). Zero is an explicit valid capacity. Missing routing or facility capacity
stops the report rather than treating the constraint as unlimited. Facility
limits carry forward from their effective dates, including shutdown zeros.
Field/QA summary rows in the long facility table are ignored.

SW routing is the default. PWRI priority assigns each well once to its PWRI
manifold if present, otherwise SW. It assumes PM01 corresponds to CPS SM and
uses PWRI1 for CPS1/4/5, PWRI2 for CPS7; confirm this mapping for your network.

Forecast: choose well baseline, start/end dates, decline percentage and annual
or monthly effective basis. The default is 6% annual effective decline:
BMT(t) = baseline BMT * 0.94 ** (days since forecast start / 365.25).
No decline is applied before forecast start. Targets, efficiency and routing
remain fixed at baseline; facility limits vary through time. Each well declines
separately, and constraints are recalculated every day. The period average is
the mean of all daily results, including both start and end dates. The well
audit in the forecast export is the end-date result. Forecast periods are
limited to 10 years.

The app runs locally and does not send uploaded well data to an external service.
