import sqlite3
import pandas as pd

def connect_to_database(db_path):
    """Établit une connexion à la base de données SQLite"""
    try:
        conn = sqlite3.connect(db_path)
        print("Connexion à la base de données réussie.")
        return conn
    except sqlite3.Error as e:
        print(f"Erreur de connexion à la base de données: {e}")
        return None

def question1_top5_clients(conn):
    query = """
    SELECT 
        c.CustomerId,
        c.FirstName || ' ' || c.LastName AS CustomerName,
        c.Country,
        ROUND(SUM(i.Total), 2) AS TotalSpent
    FROM 
        Customer c
    JOIN Invoice i ON c.CustomerId = i.CustomerId
    GROUP BY 
        c.CustomerId, c.FirstName, c.LastName, c.Country
    ORDER BY 
        TotalSpent DESC
    LIMIT 5;
    """
    
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))

def question2_top_genres_revenus(conn):
    query = """
    SELECT 
        g.Name AS Genre,
        ROUND(SUM(il.Quantity * il.UnitPrice), 2) AS Revenue,
        COUNT(DISTINCT t.TrackId) AS NumberOfTracks
    FROM 
        Genre g
    JOIN Track t ON g.GenreId = t.GenreId
    JOIN InvoiceLine il ON t.TrackId = il.TrackId
    GROUP BY 
        g.Name
    ORDER BY 
        Revenue DESC;
    """
    
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))

def question3_avg_rock_duration(conn):
    query = """
    SELECT 
        ROUND(AVG(t.Milliseconds) / 60000.0, 2) AS AvgDurationMinutes
    FROM 
        Track t
    JOIN Genre g ON t.GenreId = g.GenreId
    WHERE 
        g.Name = 'Rock';
    """
    
    df = pd.read_sql_query(query, conn)
    avg_minutes = df.iloc[0, 0]
    print(f"Durée moyenne d'un morceau de Rock: {avg_minutes} minutes")

def question4_top_employee_revenue(conn):
    query = """
    SELECT 
        e.EmployeeId,
        e.LastName || ' ' || e.FirstName AS EmployeeName,
        e.Title,
        ROUND(SUM(i.Total), 2) AS TotalRevenue
    FROM 
        Employee e
    JOIN Customer c ON e.EmployeeId = c.SupportRepId
    JOIN Invoice i ON c.CustomerId = i.CustomerId
    WHERE 
        e.Title = 'Sales Support Agent'
    GROUP BY 
        e.EmployeeId, e.LastName, e.FirstName, e.Title
    ORDER BY 
        TotalRevenue DESC
    LIMIT 1;
    """
    
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))

def main():
    db_path = "Chinook_Sqlite.sqlite"
    conn = connect_to_database(db_path)
    if conn is None:
        return
    question1_top5_clients(conn)
    question2_top_genres_revenus(conn)
    question3_avg_rock_duration(conn)
    question4_top_employee_revenue(conn)
    conn.close()

if __name__ == "__main__":
    main()