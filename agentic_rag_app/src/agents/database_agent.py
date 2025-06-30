"""
Database Agent

Natural language to SQL conversion based on examples/text_to_sql.py
"""

import sqlite3
import json
from typing import Dict, Any, List, Optional
from smolagents import Tool, ToolCallingAgent, LiteLLMModel


class SQLExecutorTool(Tool):
    """Tool for executing SQL queries safely"""
    
    name = "sql_executor"
    description = "Execute SQL queries on connected databases"
    inputs = {
        "sql_query": {
            "type": "string",
            "description": "SQL query to execute"
        },
        "database": {
            "type": "string", 
            "description": "Database name or connection",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, db_path: str = ":memory:", **kwargs):
        super().__init__(**kwargs)
        self.db_path = db_path
        self._setup_demo_data()
    
    def _setup_demo_data(self):
        """Setup demo database with sample data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sample tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER,
            department TEXT
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product TEXT,
            amount DECIMAL(10,2),
            order_date DATE,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """)
        
        # Sample data
        users_data = [
            (1, 'Alice Johnson', 'alice@company.com', 28, 'Engineering'),
            (2, 'Bob Smith', 'bob@company.com', 34, 'Sales'),
            (3, 'Carol Davis', 'carol@company.com', 31, 'Marketing'),
            (4, 'David Wilson', 'david@company.com', 29, 'Engineering'),
            (5, 'Eve Brown', 'eve@company.com', 26, 'Design')
        ]
        
        orders_data = [
            (1, 1, 'Laptop', 1299.99, '2024-01-15'),
            (2, 2, 'Monitor', 299.99, '2024-01-16'),
            (3, 1, 'Keyboard', 79.99, '2024-01-17'),
            (4, 3, 'Mouse', 49.99, '2024-01-18'),
            (5, 4, 'Headphones', 199.99, '2024-01-19'),
            (6, 2, 'Webcam', 89.99, '2024-01-20')
        ]
        
        cursor.executemany("INSERT OR REPLACE INTO users VALUES (?, ?, ?, ?, ?)", users_data)
        cursor.executemany("INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?)", orders_data)
        
        conn.commit()
        conn.close()
    
    def forward(self, sql_query: str, database: str = None) -> str:
        """Execute SQL query safely"""
        try:
            # Basic SQL injection protection
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
            query_upper = sql_query.upper()
            
            # Allow only SELECT statements for safety
            if not query_upper.strip().startswith('SELECT'):
                return "Error: Only SELECT queries are allowed for safety"
            
            # Check for dangerous keywords in SELECT queries
            for keyword in dangerous_keywords:
                if keyword in query_upper and keyword != 'SELECT':
                    return f"Error: '{keyword}' operations not allowed"
            
            # Execute query
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(sql_query)
            results = cursor.fetchall()
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            conn.close()
            
            # Format results
            if not results:
                return "Query executed successfully but returned no results"
            
            # Create formatted output
            output = f"Query Results ({len(results)} rows):\n\n"
            output += " | ".join(column_names) + "\n"
            output += "-" * (len(" | ".join(column_names))) + "\n"
            
            for row in results[:10]:  # Limit to 10 rows for display
                output += " | ".join(str(cell) for cell in row) + "\n"
            
            if len(results) > 10:
                output += f"\n... and {len(results) - 10} more rows"
            
            return output
            
        except sqlite3.Error as e:
            return f"SQL Error: {str(e)}"
        except Exception as e:
            return f"Error executing query: {str(e)}"


class SchemaInspectorTool(Tool):
    """Tool for inspecting database schema"""
    
    name = "schema_inspector"
    description = "Inspect database tables and schema"
    inputs = {
        "action": {
            "type": "string",
            "description": "Action: 'list_tables', 'describe_table', or 'show_sample'"
        },
        "table_name": {
            "type": "string",
            "description": "Table name for describe_table or show_sample actions",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, db_path: str = ":memory:", **kwargs):
        super().__init__(**kwargs)
        self.db_path = db_path
    
    def forward(self, action: str, table_name: str = None) -> str:
        """Inspect database schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if action == "list_tables":
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                table_list = [table[0] for table in tables]
                return f"Available tables: {', '.join(table_list)}"
            
            elif action == "describe_table":
                if not table_name:
                    return "Error: table_name required for describe_table action"
                
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                if not columns:
                    return f"Table '{table_name}' not found"
                
                schema = f"Schema for table '{table_name}':\n\n"
                schema += "Column | Type | Not Null | Default | Primary Key\n"
                schema += "-" * 50 + "\n"
                
                for col in columns:
                    schema += f"{col[1]} | {col[2]} | {'Yes' if col[3] else 'No'} | {col[4] or 'None'} | {'Yes' if col[5] else 'No'}\n"
                
                return schema
            
            elif action == "show_sample":
                if not table_name:
                    return "Error: table_name required for show_sample action"
                
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                results = cursor.fetchall()
                
                if not results:
                    return f"Table '{table_name}' is empty"
                
                # Get column names
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                sample = f"Sample data from '{table_name}' (first 3 rows):\n\n"
                sample += " | ".join(column_names) + "\n"
                sample += "-" * (len(" | ".join(column_names))) + "\n"
                
                for row in results:
                    sample += " | ".join(str(cell) for cell in row) + "\n"
                
                return sample
            
            else:
                return f"Unknown action: {action}. Use 'list_tables', 'describe_table', or 'show_sample'"
            
        except sqlite3.Error as e:
            return f"Database Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            conn.close()


class DatabaseAgent:
    """Database agent for natural language to SQL conversion"""
    
    def __init__(
        self, 
        db_path: str = ":memory:",
        model_id: str = "openrouter/mistralai/mistral-small-3.2-24b-instruct"
    ):
        """Initialize database agent"""
        self.db_path = db_path
        
        # Create tools
        self.sql_tool = SQLExecutorTool(db_path)
        self.schema_tool = SchemaInspectorTool(db_path)
        
        # Create model
        self.model = LiteLLMModel(
            model_id=model_id,
            temperature=0.1,
            max_tokens=1000
        )
        
        # Create agent
        self.agent = ToolCallingAgent(
            tools=[self.sql_tool, self.schema_tool],
            model=self.model,
            max_steps=6,
            planning_interval=2,
            provide_run_summary=True
        )
    
    def query(self, natural_language_query: str) -> str:
        """Convert natural language to SQL and execute"""
        
        system_prompt = """You are a Database Agent that converts natural language queries to SQL.

Your tools:
- schema_inspector: Use to explore database structure
- sql_executor: Use to run SELECT queries

Process:
1. First understand what data the user wants
2. Use schema_inspector to explore available tables and columns
3. Generate appropriate SQL SELECT query
4. Execute the query using sql_executor
5. Interpret and summarize the results for the user

Important:
- Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP)
- Always inspect schema first if you're unsure about table structure
- Provide clear explanations of what the query does
- If query fails, explain why and suggest alternatives

Example workflow:
1. User asks: "Show me all users in the Engineering department"
2. Use schema_inspector to see user table structure
3. Generate SQL: SELECT * FROM users WHERE department = 'Engineering'
4. Execute and explain results"""
        
        enhanced_query = f"{system_prompt}\n\nUser Question: {natural_language_query}"
        
        try:
            result = self.agent.run(enhanced_query)
            return str(result)
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def get_schema_info(self) -> str:
        """Get database schema information"""
        return self.schema_tool.forward("list_tables")
    
    def describe_table(self, table_name: str) -> str:
        """Describe a specific table"""
        return self.schema_tool.forward("describe_table", table_name)


def test_database_agent():
    """Test the database agent"""
    print("🗃️  DATABASE AGENT TEST")
    print("=" * 40)
    
    # Create agent
    db_agent = DatabaseAgent()
    
    # Show available tables
    print("📊 Available Tables:")
    print(db_agent.get_schema_info())
    
    # Test queries
    test_queries = [
        "Show me all users",
        "How many orders were placed?", 
        "Which users are in the Engineering department?",
        "What's the total amount of all orders?",
        "Show me the most expensive order"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        print("-" * 30)
        
        try:
            response = db_agent.query(query)
            print(f"📝 Response: {response[:300]}...")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_database_agent()