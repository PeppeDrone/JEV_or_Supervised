# ppd_bangladesh

- Observations: 800
- Predictors: 40
- Outcome: ppd_high
- Class counts: {'EPDS not high': 450, 'EPDS high risk': 350}
- Excluded: ['Angry after latest child birth', 'EPDS Score', 'Feeling about motherhood', 'Feeling bad about yourself or that you are a failure or have let yourself or your family down', 'Feeling down, depressed, or hopeless', 'Feeling for regular activities', 'Feeling tired or having little energy', 'Little interest or pleasure in doing things', 'Moving or speaking or restlessness', 'PHQ9 Result', 'PHQ9 Score', 'Poor appetite or overeating', 'Relax/sleep when newborn is tended ', 'Relax/sleep when the newborn is asleep', 'The thought of harming yourself has occurred ', 'Things have been getting to you', 'Thoughts that you would be better off dead, or of hurting yourself', 'Trouble falling or staying asleep, or sleeping too much', 'Worry about newborn', 'You have been able to laugh and see the funny side of things', 'You have been anxious or worried for no good reason', 'You have been so unhappy that you have been crying', 'You have been so unhappy that you have had difficulty sleeping', 'You have blamed myself unnecessarily when things went wrong', 'You have felt sad or miserable', 'You have felt scared or panicky for no good reason', 'You have looked forward with enjoyment to things', 'rouble concentrating on things']
- Note: Mendeley 4nznnrk8cg v3 (Raisa & Kaiser). Y = EPDS Result High (score 13–30). Current EPDS/PHQ-9 items and postpartum mood/sleep items excluded.

## Codebook review (2026-10-24)

- Source documentation: Mendeley Data 4nznnrk8cg v3 (Raisa & Kaiser), PPD_Data_Dictionary_v3.csv
- Changes: labels clarified against the data dictionary (PHQ-2 screens, delivery mode, pregnancy loss, substance use); EPDS >= 13 threshold placed in classes/question; setting rewritten.
- Validation: `jevtest check` passes; zero-shot prompt about 538 tokens; ICL subsample of about 218 rows at ~144.0 tokens per row.
