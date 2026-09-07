"""CAP master codebook (21 major topics) and corpus specifications.

Comparative Agendas Project major topic codes. Codes 11 and 22 are not used in the
master scheme; 99 ("other"), -555 (missing) and media-specific extensions (24, 26-34,
0, 91) are dropped so that every corpus shares an identical 21-way label space.
"""

CAP_MAJOR_TOPICS = {
    1:  "Macroeconomics",
    2:  "Civil Rights, Minority Issues, and Civil Liberties",
    3:  "Health",
    4:  "Agriculture",
    5:  "Labor and Employment",
    6:  "Education",
    7:  "Environment",
    8:  "Energy",
    9:  "Immigration",
    10: "Transportation",
    12: "Law, Crime, and Family Issues",
    13: "Social Welfare",
    14: "Community Development and Housing Issues",
    15: "Banking, Finance, and Domestic Commerce",
    16: "Defense",
    17: "Space, Science, Technology and Communications",
    18: "Foreign Trade",
    19: "International Affairs and Foreign Aid",
    20: "Government Operations",
    21: "Public Lands and Water Management",
    23: "Cultural Policy Issues",
}

VALID_CODES = sorted(CAP_MAJOR_TOPICS)          # 21 codes
CODE_TO_IDX = {c: i for i, c in enumerate(VALID_CODES)}

# corpus key -> (csv file, text column, label column, country, venue, human-readable name)
CORPORA = {
    "us_bills":  ("us_bills.csv",  "description", "majortopic", "US", "legislative",
                  "U.S. Congressional bill titles"),
    "us_nyt":    ("us_nyt.csv",    "title",       "majortopic", "US", "media",
                  "New York Times front-page headlines"),
    "us_eo":     ("us_eo.csv",     "description", "majortopic", "US", "executive",
                  "U.S. executive orders"),
    "us_scotus": ("us_scotus.csv", "summary",     "majortopic", "US", "judicial",
                  "U.S. Supreme Court case summaries"),
    "uk_acts":   ("uk_acts.csv",   "long_title",  "majortopic", "UK", "legislative",
                  "UK Acts of Parliament long titles"),
    "uk_pmqs":   ("uk_pmqs.csv",   "question",    "majortopic", "UK", "legislative_oral",
                  "UK Prime Minister's Questions"),
    "uk_media":  ("uk_media.csv",  "description", "majortopic", "UK", "media",
                  "UK newspaper front-page headlines"),
}

SEED = 20260905
