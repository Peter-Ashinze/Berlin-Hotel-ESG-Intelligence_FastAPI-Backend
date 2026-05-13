from fastapi import FastAPI
import pandas as pd
import numpy as np
from database import engine

app = FastAPI(
    title="Berlin Hotel ESG Intelligence API",
    description="API for Berlin hotel ESG scores, carbon emissions, AI trust risk, zones, and recommendations",
    version="1.0"
)

SUMMARY_COLUMNS = [
    "hotel_id",
    "name",
    "latitude",
    "longitude",
    "berlin_zone",
    "borough",
    "hotel_class",
    "esg_score",
    "esg_class",
    "co2_per_occupied_room_night",
    "confidence_score",
    "confidence_class",
    "final_ai_trust_risk",
    "trusted_recommendation_score",
    "trusted_rank",
]


def load_hotels():

    df = pd.read_sql(
        "SELECT * FROM berlin_hotels_esg",
        engine
    )

    # Replace infinity with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Convert all columns to object, then replace NaN with None
    df = df.astype(object).where(pd.notnull(df), None)

    return df


def safe_summary(df):

    available_cols = [
        col for col in SUMMARY_COLUMNS
        if col in df.columns
    ]

    return df[available_cols].to_dict(orient="records")


@app.get("/")
def home():

    return {
        "message": "Berlin Hotel ESG Intelligence API is running"
    }


@app.get("/hotels")
def get_hotels():

    df = load_hotels()

    return safe_summary(df)


@app.get("/hotels/top")
def get_top_hotels():

    df = load_hotels()

    top_hotels = df.sort_values(
        by="trusted_recommendation_score",
        ascending=False
    ).head(3)

    return safe_summary(top_hotels)


@app.get("/metadata/zones")
def get_zones():

    df = load_hotels()

    return sorted(
        df["berlin_zone"]
        .dropna()
        .unique()
        .tolist()
    )


@app.get("/metadata/risks")
def get_risks():

    df = load_hotels()

    return sorted(
        df["final_ai_trust_risk"]
        .dropna()
        .unique()
        .tolist()
    )


@app.get("/hotels/by-zone/{zone}")
def get_hotels_by_zone(zone: str):

    df = load_hotels()

    hotels = df[
        df["berlin_zone"]
        .astype(str)
        .str.lower()
        .str.strip()
        ==
        zone.lower().strip()
    ]

    return safe_summary(hotels)


@app.get("/hotels/by-risk/{risk}")
def get_hotels_by_risk(risk: str):

    df = load_hotels()

    hotels = df[
        df["final_ai_trust_risk"]
        .astype(str)
        .str.lower()
        .str.strip()
        ==
        risk.lower().strip()
    ]

    return safe_summary(hotels)


@app.get("/hotels/search/{search_term}")
def search_hotels(search_term: str):

    df = load_hotels()

    matches = df[
        df["name"]
        .astype(str)
        .str.lower()
        .str.contains(
            search_term.lower().strip(),
            na=False
        )
    ]

    if matches.empty:

        return {
            "message": "No matching hotels found"
        }

    return safe_summary(matches)


@app.get("/hotels/name/{hotel_name}")
def get_hotel_by_name(hotel_name: str):

    df = load_hotels()

    exact_match = df[
        df["name"]
        .astype(str)
        .str.lower()
        .str.strip()
        ==
        hotel_name.lower().strip()
    ]

    if not exact_match.empty:

        return exact_match.to_dict(
            orient="records"
        )[0]

    partial_matches = df[
        df["name"]
        .astype(str)
        .str.lower()
        .str.contains(
            hotel_name.lower().strip(),
            na=False
        )
    ]

    if partial_matches.empty:

        return {
            "message": "Hotel not found"
        }

    return {
        "message": (
            "Exact hotel name not found, "
            "but partial matches were found"
        ),
        "matches": safe_summary(
            partial_matches
        )
    }


@app.get("/recommendation/{hotel_name}")
def get_recommendation(hotel_name: str):

    df = load_hotels()

    matches = df[
        df["name"]
        .astype(str)
        .str.lower()
        .str.contains(
            hotel_name.lower().strip(),
            na=False
        )
    ]

    if matches.empty:

        return {
            "message": "Hotel not found"
        }

    selected_hotel = matches.iloc[0]

    selected_zone = (
        selected_hotel["berlin_zone"]
    )

    better_options = df[
        (df["berlin_zone"] == selected_zone)
        &
        (
            df["name"]
            !=
            selected_hotel["name"]
        )
        &
        (
            df[
                "trusted_recommendation_score"
            ]
            >
            selected_hotel[
                "trusted_recommendation_score"
            ]
        )
    ].copy()

    if better_options.empty:

        return {
            "selected_hotel":
                selected_hotel["name"],

            "berlin_zone":
                selected_zone,

            "message":
                (
                    f"No better trusted ESG "
                    f"alternative found for "
                    f"{selected_hotel['name']} "
                    f"in {selected_zone}."
                )
        }

    best_alternative = (
        better_options.sort_values(
            by="trusted_recommendation_score",
            ascending=False
        ).iloc[0]
    )

    score_improvement = (
        best_alternative[
            "trusted_recommendation_score"
        ]
        -
        selected_hotel[
            "trusted_recommendation_score"
        ]
    )

    carbon_difference = (
        selected_hotel[
            "co2_per_occupied_room_night"
        ]
        -
        best_alternative[
            "co2_per_occupied_room_night"
        ]
    )

    return {

        "selected_hotel":
            selected_hotel["name"],

        "recommended_hotel":
            best_alternative["name"],

        "berlin_zone":
            selected_zone,

        "selected_score":
            round(
                selected_hotel[
                    "trusted_recommendation_score"
                ],
                2
            ),

        "recommended_score":
            round(
                best_alternative[
                    "trusted_recommendation_score"
                ],
                2
            ),

        "score_improvement":
            round(
                score_improvement,
                2
            ),

        "selected_carbon_intensity":
            round(
                selected_hotel[
                    "co2_per_occupied_room_night"
                ],
                2
            ),

        "recommended_carbon_intensity":
            round(
                best_alternative[
                    "co2_per_occupied_room_night"
                ],
                2
            ),

        "carbon_difference":
            round(
                carbon_difference,
                2
            ),

        "selected_ai_risk":
            selected_hotel[
                "final_ai_trust_risk"
            ],

        "recommended_ai_risk":
            best_alternative[
                "final_ai_trust_risk"
            ],

        "message":
            (
                f"Instead of "
                f"{selected_hotel['name']}, "
                f"choose "
                f"{best_alternative['name']} "
                f"in {selected_zone}. "
                f"It improves the trusted ESG "
                f"score by "
                f"{score_improvement:.2f} points "
                f"and changes carbon intensity "
                f"by "
                f"{carbon_difference:.2f} "
                f"kg CO₂e per occupied room night."
            )
    }
