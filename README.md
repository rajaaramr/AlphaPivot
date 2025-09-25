# AlphaPivot Trading System

AlphaPivot is an automated trading system designed to identify and execute trading opportunities in the Indian stock market (NSE). It uses a multi-pillar approach to generate trading signals, which are then used to execute trades in both futures and options markets.

## Architecture Overview

The system is designed around a modular architecture that separates concerns into distinct components, each with a specific responsibility. This makes the system easier to understand, maintain, and extend.

### Core Components

- **Pillars (`/pillars`)**: The core of the signal generation process. Each pillar represents a different aspect of market analysis (e.g., trend, momentum, quality) and contributes to a final, composite trading signal.
- **Utilities (`/utils`)**: A collection of shared modules that provide common functionalities, such as database access, configuration management, and technical analysis calculations.
- **Execution (`executor.py`)**: The main execution engine that consumes trading signals and places orders with the broker. It is responsible for risk management, order execution, and trade lifecycle management.
- **Configuration (`config.ini`)**: A single, unified configuration file that contains all the settings for the system, including database credentials, pillar weights, and risk parameters.

### Data Flow

1.  **Data Ingestion**: The system ingests market data (futures and spot) and stores it in a PostgreSQL/TimescaleDB database.
2.  **Signal Generation**: The various pillar modules analyze the market data and generate trading signals, which are also stored in the database.
3.  **Composite Scoring**: The `composite_worker_v2.py` module combines the signals from the individual pillars into a single, composite score that represents the overall trading bias.
4.  **Trade Execution**: The `executor.py` module reads the composite scores and, if they meet the required criteria, executes trades in the market.

## Getting Started

To get the system up and running, you will need to have a PostgreSQL/TimescaleDB database set up and configured.

### Prerequisites

- Python 3.8+
- PostgreSQL/TimescaleDB

### Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the database**:
    - Open the `config.ini` file.
    - In the `[postgres]` section, update the `host`, `user`, `password`, and `database` fields with your database credentials.

4.  **Run the database migrations**:
    - The database schema is managed through a series of SQL scripts. You will need to run these scripts to create the necessary tables and views.
    - The schema definitions can be found in the `db/schema` directory (once it is created).

### Running the System

The system is designed to be run as a series of scheduled tasks. The main entry points are:

- **`scheduler/run_pillars.py`**: Runs the pillar modules to generate trading signals.
- **`executor.py`**: Runs the trade execution engine.

You can run these scripts manually or set them up to run on a schedule using a tool like `cron`.

## Contributing

We welcome contributions to AlphaPivot! If you would like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with a clear, descriptive commit message.
4.  Push your changes to your fork.
5.  Open a pull request with a detailed description of your changes.