from __future__ import annotations

import argparse

from carval.predict import predict_car


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Estimate a used car's market value")
    parser.add_argument("--make", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--engine-cc", type=float, default=None)
    parser.add_argument(
        "--transmission", required=True, choices=["Automatic", "Manual"]
    )
    parser.add_argument(
        "--fuel",
        required=True,
        choices=["Petrol", "Hybrid", "Diesel", "CNG", "Electric", "LPG"],
    )
    parser.add_argument("--km", type=float, required=True)
    parser.add_argument("--asking-price", type=float, default=None)
    args = parser.parse_args(argv)
    result = predict_car(
        make=args.make,
        model=args.model,
        variant=args.variant,
        year=args.year,
        engine_cc=args.engine_cc,
        transmission=args.transmission,
        fuel_type=args.fuel,
        km_driven=args.km,
        asking_price=args.asking_price,
    )
    for key, value in result.items():
        if key == "factors":
            print("factors:")
            for line in value:
                print(f"  - {line}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
