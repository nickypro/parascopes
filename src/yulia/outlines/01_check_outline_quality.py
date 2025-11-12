"""Script to check quality of generated outlines for gemma4b embeddings.

Detects: error messages, refusals, empty/short outlines, placeholders
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import re
from collections import defaultdict


class OutlineQualityChecker:
    """Check quality of generated outlines."""
    
    ERROR_PATTERNS = [
        r"errored?\s+request", r"error\s+occurred", r"failed\s+to\s+(?:generate|process|create)",
        r"request\s+failed", r"timeout\s+error", r"connection\s+error", r"api\s+error",
        r"http\s+error", r"status\s+code\s+\d{3}", r"exception\s+occurred",
    ]
    
    REFUSAL_PATTERNS = [
        r"^\s*I\s+can[''']?t\s+", r"^\s*I\s+cannot\s+", r"^\s*I\s+am\s+(?:not\s+)?able\s+to\s+",
        r"^\s*I'?m\s+unable\s+to\s+", r"^\s*I\s+don[''']?t\s+have\s+(?:access|the\s+ability)",
        r"^\s*(?:I\s+)?(?:apologize|sorry),?\s+(?:but\s+)?I\s+can[''']?t",
        r"^\s*As\s+an\s+AI", r"^\s*I'?m\s+just\s+an?\s+(?:AI|language\s+model)",
    ]
    
    
    def __init__(self):
        self.error_regex = re.compile("|".join(self.ERROR_PATTERNS), re.IGNORECASE)
        self.refusal_regex = re.compile("|".join(self.REFUSAL_PATTERNS), re.IGNORECASE | re.MULTILINE)
        # Detect same word repeated 4+ times: word word word word
        self.repetition_regex = re.compile(r'\b(\w+)(\s+\1){3,}\b', re.IGNORECASE)
    
    def check_outline(self, outline: str) -> Dict:
        """Check outline for quality issues."""
        if pd.isna(outline) or not isinstance(outline, str):
            return {'word_count': 0, 'issues': ['Empty or null outline']}
        
        word_count = len(outline.split())
        issues = []
        
        if not outline.strip():
            issues.append('Empty outline')
        elif word_count < 10:
            issues.append(f'Too short ({word_count} words)')
        
        if self.error_regex.search(outline):
            issues.append('Contains error message')
        if self.refusal_regex.search(outline):
            issues.append('Contains refusal pattern')
        if self.repetition_regex.search(outline):
            issues.append('Contains repetitive words')
        
        return {'word_count': word_count, 'issues': issues}
    
    def analyze_chunk(self, chunk_path: Path) -> Tuple[Dict, pd.DataFrame]:
        print(f"\nAnalyzing {chunk_path.name}...")
        
        try:
            df = pd.read_parquet(chunk_path)
            total = len(df)
            print(f"  Total samples: {total}")
            
            # Check all outlines
            checks = df['outline_generated'].apply(self.check_outline)
            
            # Aggregate metrics
            word_counts = checks.apply(lambda x: x['word_count'])
            issue_counts = defaultdict(int)
            samples_by_category = defaultdict(list)
            
            for idx, check in enumerate(checks):
                if check['issues']:
                    issue_counts['any'] += 1
                    category = check['issues'][0]
                    issue_counts[category] += 1
                    
                    # Store sample for top category
                    if len(samples_by_category[category]) < 3:
                        outline = df.iloc[idx]['outline_generated']
                        samples_by_category[category].append({
                            'example_id': df.iloc[idx].get('example_id', idx),
                            'issues': check['issues'],
                            'outline': outline[:150] if isinstance(outline, str) else str(outline)
                        })
            
            # Print results
            stats = {
                'chunk_name': chunk_path.name,
                'total_samples': total,
                'has_any_issue': issue_counts['any'],
                'issue_rate': issue_counts['any'] / total,
                'avg_words': word_counts.mean(),
                'median_words': word_counts.median(),
            }
            
            # Map issue types to counts
            issue_types = {
                'Empty or null outline': 'empty',
                'Contains error message': 'errors',
                'Contains refusal pattern': 'refusals',
                'Contains repetitive words': 'repetitions',
            }
            
            for issue_text, stat_key in issue_types.items():
                count = issue_counts.get(issue_text, 0)
                stats[stat_key] = count
            
            # Also count "too short" separately
            stats['too_short'] = sum(1 for issue in issue_counts.keys() if issue.startswith('Too short'))
            
            print(f"  Issues: errors={stats.get('errors', 0)}, refusals={stats.get('refusals', 0)}, "
                  f"repetitions={stats.get('repetitions', 0)}, empty={stats.get('empty', 0)}, "
                  f"too_short={stats['too_short']}")
            print(f"  Total with any issue: {stats['has_any_issue']} ({stats['issue_rate']:.1%})")
            print(f"  Avg words: {stats['avg_words']:.1f}, Median: {stats['median_words']:.0f}")
            
            # Show top error category examples
            if samples_by_category:
                top_cat, samples = max(samples_by_category.items(), key=lambda x: len(x))
                print(f"\n  Top error: '{top_cat}' ({issue_counts[top_cat]} samples)")
                for i, s in enumerate(samples, 1):
                    print(f"    Ex{i} (id={s['example_id']}): {s['outline']}{'...' if len(s['outline']) == 150 else ''}")
            
            return stats, df
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def analyze_all_chunks(data_dir: str):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Directory not found: {data_dir}")
        return
    
    parquet_files = sorted(data_path.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files")
    print("=" * 80)
    
    checker = OutlineQualityChecker()
    all_stats = [stats for f in parquet_files if (stats := checker.analyze_chunk(f)[0])]
    if not all_stats:
        return
    
    # Aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)
    
    total_samples = sum(s['total_samples'] for s in all_stats)
    totals = {
        'errors': sum(s.get('errors', 0) for s in all_stats),
        'refusals': sum(s.get('refusals', 0) for s in all_stats),
        'repetitions': sum(s.get('repetitions', 0) for s in all_stats),
        'empty': sum(s.get('empty', 0) for s in all_stats),
        'too_short': sum(s.get('too_short', 0) for s in all_stats),
        'any_issue': sum(s['has_any_issue'] for s in all_stats),
    }
    
    print(f"\nTotal samples: {total_samples:,}")
    print(f"\nIssue breakdown:")
    for issue_type, count in totals.items():
        if issue_type != 'any_issue':
            print(f"  - {issue_type.replace('_', ' ').title()}: {count:,} ({count/total_samples:.2%})")
    print(f"  - Total with any issue: {totals['any_issue']:,} ({totals['any_issue']/total_samples:.2%})")
    
    avg_words = sum(s['avg_words'] * s['total_samples'] for s in all_stats) / total_samples
    print(f"\nAverage outline length: {avg_words:.1f} words")
    
    # Quality assessment
    print("\n" + "=" * 80)
    print("QUALITY ASSESSMENT")
    print("=" * 80)
    
    issue_rate = totals['any_issue'] / total_samples
    quality = (
        "EXCELLENT" if issue_rate < 0.01 else
        "GOOD" if issue_rate < 0.05 else
        "FAIR" if issue_rate < 0.10 else
        "POOR"
    )
    
    print(f"\nOverall Quality: {quality} (issue rate: {issue_rate:.2%})")
    
    if totals['errors'] > 0:
        print(f"WARNING: {totals['errors']} outlines contain error messages")
    if totals['refusals'] > 0:
        print(f"WARNING: {totals['refusals']} outlines contain refusal patterns")
    if totals['repetitions'] > 0:
        print(f"WARNING: {totals['repetitions']} outlines contain repetitive words")


def main():
    """Main function."""
    import argparse
    
    default_dir = os.environ.get(
        "OUTLINE_DATA_DIR",
        str(Path(__file__).parent / "results" / "fineweb-gemma4b" / "v0.0")
    )
    
    parser = argparse.ArgumentParser(description="Check quality of generated outlines")
    parser.add_argument("--data-dir", default=default_dir,
                       help="Directory with parquet files (default: $OUTLINE_DATA_DIR or ./results/fineweb-gemma4b/v0.0)")
    
    analyze_all_chunks(parser.parse_args().data_dir)


if __name__ == "__main__":
    main()
