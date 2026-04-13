# math_server.py
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("MathServer")

@mcp.tool()
def calculate_salary_after_tax(gross_salary: float, tax_rate: float = 0.2) -> float:
    """Calculates net salary after applying a tax rate."""
    return gross_salary * (1 - tax_rate)

@mcp.tool()
def get_current_stock_price(ticker: str) -> str:
    """A mock tool to simulate fetching stock prices."""
    return f"The current price for {ticker} is $150.00"

if __name__ == "__main__":
    mcp.run(transport="stdio")