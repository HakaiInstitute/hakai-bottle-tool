from hakai_bottle_tool.hakai_bottle_tool import get_bottle_data,save_bottle_to
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join CTD vs Bottle Sample data")
    parser.add_argument("-station", type=str, nargs="+", help="Station to review")
    parser.add_argument(
        "-time_min",
        help="Minimum Time in pandas.to_datetime compatible format",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-time_max",
        help="Maximum Time in pandas.to_datetime compatible format",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-format",
        help="File format to output ('csv' or 'netcdf' [default])",
        type=str,
        default="netcdf",
    )
    parser.add_argument(
        "-output_path",
        help="File output path",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    print(
        f"Get data from {args.station[0]} from {args.time_min or '...'} to {args.time_max or '...'}"
    )
    df = get_bottle_data(args.station[0], args.time_min, args.time_max)

    # Save to file format
    save_bottle_to(df, args.station[0], args.output_path, args.format)