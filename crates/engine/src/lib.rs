use pyo3::prelude::*;

mod types;
mod orderbook;
mod account;
mod exchange;

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

    // Enums
    m.add_class::<types::BookAction>()?;
    m.add_class::<types::OrderSide>()?;
    m.add_class::<types::OrderStatus>()?;
    m.add_class::<types::AggressorSide>()?;
    m.add_class::<types::TimeInForce>()?;

    // Data structs
    m.add_class::<types::Instrument>()?;
    m.add_class::<types::FairValueData>()?;
    m.add_class::<types::OrderBookDelta>()?;
    m.add_class::<types::TradeTick>()?;
    m.add_class::<types::Order>()?;
    m.add_class::<types::Fill>()?;
    m.add_class::<types::Position>()?;

    // Engine core (only available as PyClass when not in test mode)
    #[cfg(not(test))]
    m.add_class::<exchange::EngineCore>()?;

    // Flag constants
    m.add("F_SNAPSHOT", types::F_SNAPSHOT)?;
    m.add("F_LAST", types::F_LAST)?;

    Ok(())
}
