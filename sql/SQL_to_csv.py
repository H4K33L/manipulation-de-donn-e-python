import sqlite3
import pandas as pd

def connect_to_database(db_path):
    try:
        conn = sqlite3.connect(db_path)
        print("Connexion à la base de données réussie.")
        return conn
    except sqlite3.Error as e:
        print(f"Erreur de connexion à la base de données: {e}")
        return None

def fetch_merged_data(conn):
    query = """
    SELECT 
        c.CustomerId,
        c.FirstName || ' ' || c.LastName AS CustomerName,
        c.Country,
        e.EmployeeId,
        e.LastName || ' ' || e.FirstName AS EmployeeName,
        SUM(i.Total) AS TotalSpent,
        COUNT(DISTINCT i.InvoiceId) AS NumberOfPurchases,
        GROUP_CONCAT(DISTINCT g.Name) AS GenresPurchased,
        AVG(t.Milliseconds)/60000.0 AS AvgTrackDurationMinutes,
        COUNT(DISTINCT t.TrackId) AS UniqueTracksPurchased
    FROM 
        Customer c
    LEFT JOIN Invoice i ON c.CustomerId = i.CustomerId
    LEFT JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
    LEFT JOIN Track t ON il.TrackId = t.TrackId
    LEFT JOIN Genre g ON t.GenreId = g.GenreId
    LEFT JOIN Employee e ON c.SupportRepId = e.EmployeeId
    GROUP BY 
        c.CustomerId, c.FirstName, c.LastName, c.Country, e.EmployeeId, e.LastName, e.FirstName
    ORDER BY 
        TotalSpent DESC;
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        print("Données fusionnées récupérées avec succès.")
        return df
    except Exception as e:
        print(f"Erreur lors de l'exécution de la requête: {e}")
        return None

def export_to_csv(df, output_file):
    try:
        df.to_csv(output_file, index=False)
        print(f"Données exportées avec succès dans {output_file}")
    except Exception as e:
        print(f"Erreur lors de l'export vers CSV: {e}")

def main():
    db_path = "Chinook_Sqlite.sqlite"
    output_csv = "chinook_merged_data.csv"
    conn = connect_to_database(db_path)
    if conn is None:
        return
    merged_data = fetch_merged_data(conn)
    if merged_data is not None:
        print("\nAperçu des données fusionnées:")
        print(merged_data.head())
        export_to_csv(merged_data, output_csv)
    
    conn.close()
    print("\nConnexion à la base de données fermée.")

if __name__ == "__main__":
    main()